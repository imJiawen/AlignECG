from numpy import ones_like
import torch
import torch.nn as nn
import warnings
import math
import torch.nn.functional as F
from einops import rearrange, repeat
import sys
import os

from libcpab import Cpab

###############
### HELPERS ###
###############

def reset_parameters(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()

class Constant(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        self.output_sizes = output_sizes
        self.const = nn.parameter.Parameter(torch.Tensor(1, *output_sizes))

    # inp is an arbitrary tensor, whose values will be ignored;
    # output is self.const expanded over the first dimension of inp.
    # output.shape = (inp.shape[0], *output_sizes)
    def forward(self, inp):
        return self.const.expand(inp.shape[0], *((-1,)*len(self.output_sizes)))

    def reset_parameters(self):
        nn.init.uniform_(self.const, -1, 1) # U~[-1,1]

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim
    def forward(self, inp):
        return inp.unsqueeze(self._dim)

class Square(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return inp*inp

class Abs(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.abs(inp)

class Exp(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.exp(inp)

class Sin(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.sin(inp)
    
#########################
### Localization net ###
#########################

def get_locnet():
    # Spatial transformer localization-network
    locnet = nn.Sequential(
        nn.Conv1d(1, 128, kernel_size=7),
        # nn.BatchNorm1d(128),
        nn.MaxPool1d(3, stride=2),
        nn.ReLU(True),
        nn.Conv1d(128, 64, kernel_size=9),
        # nn.BatchNorm1d(64),
        nn.MaxPool1d(3, stride=3),
        nn.ReLU(True),
        nn.Conv1d(64, 64, kernel_size=3),
        # nn.BatchNorm1d(),
        nn.MaxPool1d(3, stride=2),
        nn.ReLU(True),
        # GAP (when size=1) -
        # Note: While GAP allow the model size to remain fix w.r.t input length,
        # Temporal information is lost by the GAP operator.
        #nn.AdaptiveAvgPool1d(1),
    )
    return locnet


class Scoring_Layer(nn.Module):
    def __init__(self, d_model):
        super(Scoring_Layer, self).__init__()
        self.score_net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model,1, bias=False))

    def forward(self, h):
        # input: x: [b 1 l], mask: [b 1 l]
        # output: score: [b, l]
        h = rearrange(h, 'b 1 l -> b l 1')
        h = self.score_net(h).squeeze(-1) # [b l]
        return h

#########################
### WARPING FUNCTIONS ###
#########################


class CPABWarp(nn.Module):
    # loc_net can be any nn.Module that takes shape (batch_size, seq_len, input_size)
    # and produces shape (batch_size, n_cells-1), where n_cells is the number of cells in the tessellation
    def __init__(self, n_cells, loc_net):
        super().__init__()
        if not isinstance(loc_net, nn.Module):
            raise ValueError("loc_net must be an instance of torch.nn.Module")

        self.cpab = Cpab([n_cells], "pytorch", "gpu", zero_boundary=True, volume_perservation=False)
        self.loc_net = loc_net

    # input_seq.shape = (batch_size, seq_len, input_size)
    # output shape = (batch_size, seq_len)
    def forward(self, input_seq, mask):
        mask = rearrange(mask, 'b 1 l -> b l')
        batch_size, _, seq_len = input_seq.shape
        # grid = self.cpab.uniform_meshgrid([seq_len]) # [1,l]
        grid = self.cpab.uniform_meshgrid((seq_len,))
        theta = self.loc_net(input_seq) # [b l]
        gamma = self.cpab.transform_grid(grid, theta)
        gamma = gamma.reshape(batch_size, seq_len)
        return gamma


# backend can be any nn.Module that takes shape (batch_size, seq_len, input_size)
# and produces shape (batch_size, seq_len); the output of the backend is normalized
# and integrated.
class VanillaWarp(nn.Module):
    def __init__(self, backend, nonneg_trans='sigmoid'):
        super().__init__()
        if not isinstance(backend, nn.Module):
            raise ValueError("backend must be an instance of torch.nn.Module")
        self.backend = backend
        self.normintegral = NormalizedIntegral(nonneg_trans)

    # input_seq.shape = (b k l d)
    # output shape = (batch_size, seq_len)
    def forward(self, input_seq, mask):
        score = self.backend(input_seq) # [b 1 l] -> [b l]
        gamma = self.normintegral(score, mask) 
        return gamma

class NormalizedIntegral(nn.Module):
    # {abs, square, relu}      -> warping variance more robust to input variance
    # {exp, softplus, sigmoid} -> warping variance increases with input variance, strongest for exp
    def __init__(self, nonneg):
        super().__init__()
        # higher warping variance
        if nonneg == 'square':
            self.nonnegativity = Square()
        elif nonneg == 'relu':
            warnings.warn('ReLU non-negativity does not necessarily result in a strictly monotonic warping function gamma! In the worst case, gamma == 0 everywhere.', RuntimeWarning)
            self.nonnegativity = nn.ReLU()
        elif nonneg == 'exp':
            self.nonnegativity = Exp()
        # lower warping variance
        elif nonneg == 'abs':
            self.nonnegativity = Abs()
        elif nonneg == 'sigmoid':
            self.nonnegativity = nn.Sigmoid()
        elif nonneg == 'softplus':
            self.nonnegativity = nn.Softplus()
        else:
            raise ValueError("unknown non-negativity transformation, try: abs, square, exp, relu, softplus, sigmoid")

    # input_seq.shape = (batch_size, seq_len)
    # output shape    = (batch_size, seq_len)
    def forward(self, input_seq, mask):
        gamma = self.nonnegativity(input_seq)
        mask = mask.squeeze(1)
        # transform sequences to alignment functions between 0 and 1
        # dgamma = torch.cat([torch.zeros((gamma.shape[0],1)).to(input_seq.device), gamma], dim=1) # fix entry to 0
        mask_mask = torch.ones(gamma.shape).to(input_seq.device)
        # mask = rearrange(mask, 'b k l -> (b k) l')
        mask_mask[:,0] = 0
        mask = mask * mask_mask
        dgamma = mask * gamma
        gamma = torch.cumsum(dgamma, dim=-1) * mask
        # gamma /= torch.max(gamma, dim=1)[0].unsqueeze(1)
        gamma_max = torch.max(gamma, dim=1)[0].unsqueeze(1)
        gamma_max[gamma_max==0] = 1
        gamma = gamma / gamma_max
        return gamma

##########################
### WARPING LAYER ###
##########################

class Almtx(nn.Module):
    def __init__(self, signal_len, channels, K, device,  d_model=64, warp='cpa'):
        super().__init__()
        self.S = K
        self.signal_len = signal_len
        self.warp_name = warp
        if warp =='cusum':
            loc_net = Scoring_Layer(d_model).to(device)
            # loc_net = Scoring_Layer(signal_len, device).to(device)
            self.warp = VanillaWarp(loc_net).to(device)
        elif warp =='cpab':
            # self.warp = CPABWarp(opt.n_cells, loc_net).to(device)
            self.warp = DTAN(signal_len, channels, device, tess=[16, ], n_recurrence=1).to(device)

        self.device = device

    def cal_new_bound(self, Rl, Rr, gamma):
        # cal new Rl
        B, S, L = gamma.shape
        mask = (Rr - gamma >= 0)
        vl, _ = torch.max(mask * gamma.detach(),-1)
        new_Rl = torch.min(vl,torch.arange(0, 1, 1/S).to(gamma.device)).unsqueeze(-1).expand(B,S,L)

        # cal new Rr
        mask = (gamma - Rl >= 0)
        mask[mask==False] = 10
        mask[mask==True] = 1
        vr, _ = torch.min(mask * gamma.detach(),-1)
        tmp_Rr = torch.max(vr,torch.arange(1/S, 1+1/S, 1/S).to(gamma.device)).unsqueeze(-1).expand(B,S,L)
        new_Rr = tmp_Rr.clone()
        new_Rr[:,-1] = tmp_Rr[:,-1] + 1e-4

        return new_Rl, new_Rr

    def forward(self, input_seq, mask):
        # [b 1 l]
        if self.warp_name == 'cpab':
            transfrmed_x, gamma = self.warp(input_seq,mask) # [B L]
        else:
            gamma = self.warp(input_seq,mask)

        mask = repeat(mask,'b 1 l -> b s l', s=self.S)
        _, L = gamma.shape
        gamma = repeat(gamma, 'b l -> b s l', s=self.S)
        Rl = torch.arange(0, 1, 1/self.S).unsqueeze(0).unsqueeze(-1).expand(1, self.S, L).to(gamma.device)
        Rr = torch.arange(1/self.S, 1+1/self.S, 1/self.S).unsqueeze(0).unsqueeze(-1).expand(1, self.S, L).to(gamma.device)

        new_Rl, new_Rr = self.cal_new_bound(Rl, Rr, gamma)
        bound_mask = mask * ((gamma - new_Rl >= 0) & (new_Rr - gamma > 0)) # [b s l]
        A = torch.threshold(gamma - new_Rl, 0, 0) + torch.threshold(new_Rr - gamma, 0, 0)

        A_diag = A * bound_mask
        A_sum = A_diag.sum(dim=-1, keepdim=True)
        A_sum = torch.where(A_sum==0, torch.ones_like(A_sum), A_sum).to(A_sum.device)
        A_norm = A_diag / A_sum

        A_norm = rearrange(A_norm, 'b s l -> b l s')
        # bound_mask = rearrange(bound_mask, 'b s l -> b l s')

        out = torch.matmul(input_seq, A_norm)
        return out, A_norm



class DTAN(nn.Module):
    '''
    PyTroch nn.Module implementation of Diffeomorphic Temporal Alignment Nets [1]
    '''
    def __init__(self, signal_len, channels, device, tess=[6, ], n_recurrence=1, zero_boundary=True):
        '''
        Args:
            signal_len (int): signal length
            channels (int): number of channels
            tess (list): tessellation shape.
            n_recurrence (int): Number of recurrences for R-DTAN. Increasing the number of recurrences
                            Does not increase the number of parameters, but does the trainning time. Default is 1.
            zero_boundary (bool): Zero boundary (when True) for input X and transformed version X_T,
                                  sets X[0]=X_T[0] and X[n] = X_T[n]. Default is true.
            device: 'gpu' or 'cpu'
        '''
        super(DTAN, self).__init__()

        # init CPAB transformer
        self.T = Cpab(tess, backend='pytorch', device='gpu', zero_boundary=zero_boundary, volume_perservation=False)
        self.dim = self.T.get_theta_dim()
        self.n_recurrence = n_recurrence
        self.input_shape = signal_len # signal len
        self.channels = channels
        self.device = device
        self.localization = get_locnet().to(device)
        self.fc_input_dim = self.get_conv_to_fc_dim()

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, self.dim),
            # Tanh constrains theta between -1 and 1
            nn.Tanh()
        ).to(device)

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[-2].bias.data.copy_(torch.clone(self.T.identity(epsilon=0.001).view(-1)))

    def get_conv_to_fc_dim(self):
        rand_tensor = torch.rand([1, self.channels, self.input_shape]).to(self.device)
        out_tensor = self.localization(rand_tensor)
        conv_to_fc_dim = out_tensor.size(1)*out_tensor.size(2)
        #print("conv_to_fc_dim",conv_to_fc_dim, "full size", out_tensor.size())
        return conv_to_fc_dim

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_input_dim)
        theta = self.fc_loc(xs)
        grid = self.T.uniform_meshgrid((self.input_shape,))
        # theta = self.loc_net(input_seq) # [b l]
        x = self.T.transform_data(x, theta, outsize=(self.input_shape,))
        theta = self.T.transform_grid(grid, theta)
        return x, theta

    def forward(self, x, mask=None):
        # transform the input
        thetas = []
        for i in range(self.n_recurrence):
            x, theta = self.stn(x)
            thetas.append(theta)
        return x, thetas[0].squeeze(1)

    def get_basis(self):
        return self.T

# References:
# [1] - Diffeomorphic Temporal Alignment Nets (NeurIPS 2019)