
# %% auto 0
__all__ = ['lin_zero_init', 'Conv', 'ConvBN', 'CoordConv', 'SepConv', 'BN1d', 'IN1d', 'Noop', 'LogitAdjLayer', 'GWAP1d',
           'pytorch_acts', 'pytorch_act_names', 'pool_head', 'average_pool_head', 'concat_pool_head', 'pool_plus_head',
           'conv_head', 'mlp_head', 'fc_head', 'rnn_head', 'conv_lin_nd_head', 'conv_lin_3d_head',
           'create_conv_lin_3d_head', 'create_lin_nd_head', 'lin_3d_head', 'create_lin_3d_head', 'conv_3d_head',
           'heads', 'test_module_to_torchscript', 'init_lin_zero', 'SwishBeta', 'SmeLU', 'Chomp1d', 'same_padding1d',
           'Pad1d', 'SameConv1d', 'same_padding2d', 'Pad2d', 'Conv2dSame', 'Conv2d', 'CausalConv1d', 'Conv1d',
           'SeparableConv1d', 'AddCoords1d', 'ConvBlock', 'ResBlock1dPlus', 'SEModule1d', 'Norm', 'LinLnDrop',
           'LambdaPlus', 'Squeeze', 'Unsqueeze', 'Add', 'Concat', 'Unfold', 'Permute', 'Transpose', 'View', 'Reshape',
           'Max', 'LastStep', 'SoftMax', 'Clamp', 'Clip', 'ReZero', 'DropPath', 'Sharpen', 'Sequential',
           'TimeDistributed', 'Temp_Scale', 'Vector_Scale', 'Matrix_Scale', 'get_calibrator', 'LogitAdjustmentLayer',
           'PPV', 'PPAuc', 'MaxPPVPool1d', 'AdaptiveWeightedAvgPool1d', 'GAP1d', 'GACP1d', 'GAWP1d',
           'GlobalWeightedAveragePool1d', 'gwa_pool_head', 'AttentionalPool1d', 'GAttP1d', 'attentional_pool_head',
           'PoolingLayer', 'GEGLU', 'ReGLU', 'get_act_fn', 'RevIN', 'create_pool_head', 'max_pool_head',
           'create_pool_plus_head', 'create_conv_head', 'create_mlp_head', 'create_fc_head', 'create_rnn_head',
           'imputation_head', 'create_conv_lin_nd_head', 'lin_nd_head', 'rocket_nd_head', 'xresnet1d_nd_head',
           'create_conv_3d_head', 'universal_pool_head', 'SqueezeExciteBlock', 'GaussianNoise',
           'PositionwiseFeedForward', 'TokenLayer', 'ScaledDotProductAttention', 'MultiheadAttention', 'MultiConv1d',
           'LSTMOutput', 'emb_sz_rule', 'TSEmbedding', 'MultiEmbedding']

# %% ../../nbs/029_models.layers.ipynb 3
from torch.jit import TracerWarning
import warnings
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn.init import normal_
from fastcore.basics import snake2camel
from fastcore.test import test_eq
from fastai.layers import *
from fastai.losses import *
from .imports import *
from .utils import *

warnings.filterwarnings("ignore", category=TracerWarning)

# %% ../../nbs/029_models.layers.ipynb 4
def test_module_to_torchscript(
    m:torch.nn.Module, # The PyTorch module to be tested.
    inputs:Tensor, # A tensor or tuple of tensors representing the inputs to the model.
    trace:bool=True, #  If `True`, attempts to trace the model. Defaults to `True`.
    script:bool=True, # If `True`, attempts to script the model. Defaults to `True`.
    serialize:bool=True, # If `True`, saves and loads the traced/scripted module to ensure it can be serialized. Defaults to `True`.
    verbose:bool=True, # If `True`, prints detailed information about the tracing and scripting process. Defaults to `True`.
):
    "Tests if a PyTorch module can be correctly traced or scripted and serialized"
    
    m = m.eval()
    m_name = m.__class__.__name__

    # Ensure inputs are in a tuple or list format
    inp_is_tuple = isinstance(inputs, (tuple, list))

    # Get the model's output
    output = m(*inputs) if inp_is_tuple else m(inputs)
    output_shapes = output.shape if not isinstance(output, (tuple, list)) else [o.shape for o in output]
    if verbose:
        print(f"output.shape: {output_shapes}")

    # Try tracing the model
    if trace:
        if verbose:
            print("Tracing...")
        try:
            traced_m = torch.jit.trace(m, inputs)
            if serialize:
                file_path = Path(f"test_traced_{m_name}.pt")
                torch.jit.save(traced_m, file_path)
                traced_mod = torch.jit.load(file_path)
                file_path.unlink()
            traced_output = traced_m(*inputs) if inp_is_tuple else traced_m(inputs)
            torch.testing.assert_close(traced_output, output)
            if verbose:
                print(f"...{m_name} has been successfully traced 😃\n")
            return True
        except Exception as e:
            if verbose:
                print(f"{m_name} cannot be traced 😔")
                print(e)
                print("\n")

    # Try scripting the model
    if script:
        if verbose:
            print("Scripting...")
        try:
            scripted_m = torch.jit.script(m)
            if serialize:
                file_path = Path(f"test_scripted_{m_name}.pt")
                torch.jit.save(scripted_m, file_path)
                scripted_mod = torch.jit.load(file_path)
                file_path.unlink()
            scripted_output = scripted_m(*inputs) if inp_is_tuple else scripted_m(inputs)
            torch.testing.assert_close(scripted_output, output)
            if verbose:
                print(f"...{m_name} has been successfully scripted 😃\n")
            return True
        except Exception as e:
            if verbose:
                print(f"{m_name} cannot be scripted 😔")
                print(e)

    return False

# %% ../../nbs/029_models.layers.ipynb 6
def init_lin_zero(m):
    if isinstance(m, (nn.Linear)): 
        if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 0)
    for l in m.children(): init_lin_zero(l)
        
lin_zero_init = init_lin_zero

# %% ../../nbs/029_models.layers.ipynb 7
class SwishBeta(Module):
    def __multiinit__(self, beta=1.): 
        self.sigmoid = torch.sigmoid
        self.beta = nn.Parameter(torch.Tensor(1).fill_(beta))
    def forward(self, x): return x.mul(self.sigmoid(x*self.beta))

# %% ../../nbs/029_models.layers.ipynb 8
class SmeLU(nn.Module):
    "Smooth ReLU activation function based on https://arxiv.org/pdf/2202.06499.pdf"

    def __init__(self, 
        beta: float = 2. # Beta value
        ) -> None:
        super().__init__()
        self.beta = abs(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.abs(x) <= self.beta, ((x + self.beta) ** 2) / (4. * self.beta), F.relu(x))

# %% ../../nbs/029_models.layers.ipynb 9
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# %% ../../nbs/029_models.layers.ipynb 10
def same_padding1d(seq_len, ks, stride=1, dilation=1):
    "Same padding formula as used in Tensorflow"
    p = (seq_len - 1) * stride + (ks - 1) * dilation + 1 - seq_len
    return p // 2, p - p // 2


class Pad1d(nn.ConstantPad1d):
    def __init__(self, padding, value=0.):
        super().__init__(padding, value)

        
# @delegates(nn.Conv1d.__init__)
class SameConv1d(Module):
    "Conv1d with padding='same'"
    def __init__(self, ni, nf, ks=3, stride=1, dilation=1, **kwargs):
        self.ks, self.stride, self.dilation = ks, stride, dilation
        self.conv1d_same = nn.Conv1d(ni, nf, ks, stride=stride, dilation=dilation, **kwargs)
        self.weight = self.conv1d_same.weight
        self.bias = self.conv1d_same.bias
        self.pad = Pad1d

    def forward(self, x):
        self.padding = same_padding1d(x.shape[-1], self.ks, dilation=self.dilation) #stride=self.stride not used in padding calculation!
        return self.conv1d_same(self.pad(self.padding)(x))

# %% ../../nbs/029_models.layers.ipynb 11
def same_padding2d(H, W, ks, stride=(1, 1), dilation=(1, 1)):
    "Same padding formula as used in Tensorflow"
    if isinstance(ks, Integral): ks = (ks, ks)
    if ks[0] == 1:  p_h = 0
    else:  p_h = (H - 1) * stride[0] + (ks[0] - 1) * dilation[0] + 1 - H
    if ks[1] == 1:  p_w = 0
    else:  p_w = (W - 1) * stride[1] + (ks[1] - 1) * dilation[1] + 1 - W
    return (p_w // 2, p_w - p_w // 2, p_h // 2, p_h - p_h // 2)


class Pad2d(nn.ConstantPad2d):
    def __init__(self, padding, value=0.):
        super().__init__(padding, value)


# @delegates(nn.Conv2d.__init__)
class Conv2dSame(Module):
    "Conv2d with padding='same'"
    def __init__(self, ni, nf, ks=(3, 3), stride=(1, 1), dilation=(1, 1), **kwargs):
        if isinstance(ks, Integral): ks = (ks, ks)
        if isinstance(stride, Integral): stride = (stride, stride)
        if isinstance(dilation, Integral): dilation = (dilation, dilation)
        self.ks, self.stride, self.dilation = ks, stride, dilation
        self.conv2d_same = nn.Conv2d(ni, nf, ks, stride=stride, dilation=dilation, **kwargs)
        self.weight = self.conv2d_same.weight
        self.bias = self.conv2d_same.bias
        self.pad = Pad2d

    def forward(self, x):
        self.padding = same_padding2d(x.shape[-2], x.shape[-1], self.ks, dilation=self.dilation) #stride=self.stride not used in padding calculation!
        return self.conv2d_same(self.pad(self.padding)(x))
    
    
# @delegates(nn.Conv2d.__init__)
def Conv2d(ni, nf, kernel_size=None, ks=None, stride=1, padding='same', dilation=1, init='auto', bias_std=0.01, **kwargs):
    "conv1d layer with padding='same', 'valid', or any integer (defaults to 'same')"
    assert not (kernel_size and ks), 'use kernel_size or ks but not both simultaneously'
    assert kernel_size is not None or ks is not None, 'you need to pass a ks'
    kernel_size = kernel_size or ks
    if padding == 'same': 
        conv = Conv2dSame(ni, nf, kernel_size, stride=stride, dilation=dilation, **kwargs)
    elif padding == 'valid': conv = nn.Conv2d(ni, nf, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs)
    else: conv = nn.Conv2d(ni, nf, kernel_size, stride=stride, padding=padding, dilation=dilation, **kwargs)
    init_linear(conv, None, init=init, bias_std=bias_std)
    return conv

# %% ../../nbs/029_models.layers.ipynb 13
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, ni, nf, ks, stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__(ni, nf, kernel_size=ks, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        self.__padding = (ks - 1) * dilation
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

# %% ../../nbs/029_models.layers.ipynb 14
# @delegates(nn.Conv1d.__init__)
def Conv1d(ni, nf, kernel_size=None, ks=None, stride=1, padding='same', dilation=1, init='auto', bias_std=0.01, **kwargs):
    "conv1d layer with padding='same', 'causal', 'valid', or any integer (defaults to 'same')"
    assert not (kernel_size and ks), 'use kernel_size or ks but not both simultaneously'
    assert kernel_size is not None or ks is not None, 'you need to pass a ks'
    kernel_size = kernel_size or ks
    if padding == 'same': 
        if kernel_size%2==1: 
            conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=kernel_size//2 * dilation, dilation=dilation, **kwargs)
        else:
            conv = SameConv1d(ni, nf, kernel_size, stride=stride, dilation=dilation, **kwargs)
    elif padding == 'causal': conv = CausalConv1d(ni, nf, kernel_size, stride=stride, dilation=dilation, **kwargs)
    elif padding == 'valid': conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs)
    else: conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=padding, dilation=dilation, **kwargs)
    init_linear(conv, None, init=init, bias_std=bias_std)
    return conv

# %% ../../nbs/029_models.layers.ipynb 21
class SeparableConv1d(Module):
    def __init__(self, ni, nf, ks, stride=1, padding='same', dilation=1, bias=True, bias_std=0.01):
        self.depthwise_conv = Conv1d(ni, ni, ks, stride=stride, padding=padding, dilation=dilation, groups=ni, bias=bias)
        self.pointwise_conv = nn.Conv1d(ni, nf, 1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        if bias:
            if bias_std != 0: 
                normal_(self.depthwise_conv.bias, 0, bias_std)
                normal_(self.pointwise_conv.bias, 0, bias_std)
            else: 
                self.depthwise_conv.bias.data.zero_()
                self.pointwise_conv.bias.data.zero_()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

# %% ../../nbs/029_models.layers.ipynb 23
class AddCoords1d(Module):
    """Add coordinates to ease position identification without modifying mean and std"""
    def forward(self, x):
        bs, _, seq_len = x.shape
        cc = torch.linspace(-1,1,x.shape[-1], device=x.device).repeat(bs, 1, 1)
        cc = (cc - cc.mean()) / cc.std()
        x = torch.cat([x, cc], dim=1)
        return x

# %% ../../nbs/029_models.layers.ipynb 25
class ConvBlock(nn.Sequential):
    "Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."
    def __init__(self, ni, nf, kernel_size=None, ks=3, stride=1, padding='same', bias=None, bias_std=0.01, norm='Batch', zero_norm=False, bn_1st=True,
                 act=nn.ReLU, act_kwargs={}, init='auto', dropout=0., xtra=None, coord=False, separable=False,  **kwargs):
        kernel_size = kernel_size or ks
        ndim = 1
        layers = [AddCoords1d()] if coord else []
        norm_type = getattr(NormType,f"{snake2camel(norm)}{'Zero' if zero_norm else ''}") if norm is not None else None
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None: bias = not (bn or inn)
        if separable: conv = SeparableConv1d(ni + coord, nf, ks=kernel_size, bias=bias, stride=stride, padding=padding, **kwargs)
        else: conv = Conv1d(ni + coord, nf, ks=kernel_size, bias=bias, stride=stride, padding=padding, **kwargs)
        act = None if act is None else act(**act_kwargs)
        if not separable: init_linear(conv, act, init=init, bias_std=bias_std)
        if   norm_type==NormType.Weight:   conv = weight_norm(conv)
        elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
        layers += [conv]
        act_bn = []        
        if act is not None: act_bn.append(act)
        if bn: act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn: act_bn.append(InstanceNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st: act_bn.reverse()
        if dropout: layers += [nn.Dropout(dropout)]
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)     
                            
Conv = named_partial('Conv', ConvBlock, norm=None, act=None)
ConvBN = named_partial('ConvBN', ConvBlock, norm='Batch', act=None)
CoordConv = named_partial('CoordConv', ConvBlock, norm=None, act=None, coord=True)
SepConv = named_partial('SepConv', ConvBlock, norm=None, act=None, separable=True)

# %% ../../nbs/029_models.layers.ipynb 26
class ResBlock1dPlus(Module):
    "Resnet block from `ni` to `nh` with `stride`"
#     @delegates(ConvLayer.__init__)
    def __init__(self, expansion, ni, nf, coord=False, stride=1, groups=1, reduction=None, nh1=None, nh2=None, dw=False, g2=1,
                 sa=False, sym=False, norm='Batch', zero_norm=True, act_cls=defaults.activation, ks=3,
                 pool=AvgPool, pool_first=True, **kwargs):
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf,ni = nf*expansion,ni*expansion
        k0 = dict(norm=norm, zero_norm=False, act=act_cls, **kwargs)
        k1 = dict(norm=norm, zero_norm=zero_norm, act=None, **kwargs)
        convpath  = [ConvBlock(ni,  nh2, ks, coord=coord, stride=stride, groups=ni if dw else groups, **k0),
                     ConvBlock(nh2,  nf, ks, coord=coord, groups=g2, **k1)
        ] if expansion == 1 else [
                     ConvBlock(ni,  nh1, 1, coord=coord, **k0),
                     ConvBlock(nh1, nh2, ks, coord=coord, stride=stride, groups=nh1 if dw else groups, **k0),
                     ConvBlock(nh2,  nf, 1, coord=coord, groups=g2, **k1)]
        if reduction: convpath.append(SEModule(nf, reduction=reduction, act_cls=act_cls))
        if sa: convpath.append(SimpleSelfAttention(nf,ks=1,sym=sym))
        self.convpath = nn.Sequential(*convpath)
        idpath = []
        if ni!=nf: idpath.append(ConvBlock(ni, nf, 1, coord=coord, act=None, **kwargs))
        if stride!=1: idpath.insert((1,0)[pool_first], pool(stride, ndim=1, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = defaults.activation(inplace=True) if act_cls is defaults.activation else act_cls()

    def forward(self, x): return self.act(self.convpath(x) + self.idpath(x))

# %% ../../nbs/029_models.layers.ipynb 27
def SEModule1d(ni, reduction=16, act=nn.ReLU, act_kwargs={}):
    "Squeeze and excitation module for 1d"
    nf = math.ceil(ni//reduction/8)*8
    assert nf != 0, 'nf cannot be 0'
    return SequentialEx(nn.AdaptiveAvgPool1d(1), 
                        ConvBlock(ni, nf, ks=1, norm=None, act=act, act_kwargs=act_kwargs),
                        ConvBlock(nf, ni, ks=1, norm=None, act=nn.Sigmoid), ProdLayer())

# %% ../../nbs/029_models.layers.ipynb 29
def Norm(nf, ndim=1, norm='Batch', zero_norm=False, init=True, **kwargs):
    "Norm layer with `nf` features and `ndim` with auto init."
    assert 1 <= ndim <= 3
    nl = getattr(nn, f"{snake2camel(norm)}Norm{ndim}d")(nf, **kwargs)
    if nl.affine and init:
        nl.bias.data.fill_(1e-3)
        nl.weight.data.fill_(0. if zero_norm else 1.)
    return nl

BN1d = partial(Norm, ndim=1, norm='Batch')
IN1d = partial(Norm, ndim=1, norm='Instance')

# %% ../../nbs/029_models.layers.ipynb 33
class LinLnDrop(nn.Sequential):
    "Module grouping `LayerNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, ln=True, p=0., act=None, lin_first=False):
        layers = [nn.LayerNorm(n_out if lin_first else n_in)] if ln else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not ln)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

# %% ../../nbs/029_models.layers.ipynb 35
class LambdaPlus(Module):
    def __init__(self, func, *args, **kwargs): self.func,self.args,self.kwargs=func,args,kwargs
    def forward(self, x): return self.func(x, *self.args, **self.kwargs)

# %% ../../nbs/029_models.layers.ipynb 36
class Squeeze(Module):
    def __init__(self, dim=-1): self.dim = dim
    def forward(self, x): return x.squeeze(dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'


class Unsqueeze(Module):
    def __init__(self, dim=-1): self.dim = dim
    def forward(self, x): return x.unsqueeze(dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'


class Add(Module):
    def forward(self, x, y): return x.add(y)
    def __repr__(self): return f'{self.__class__.__name__}'


class Concat(Module):
    def __init__(self, dim=1): self.dim = dim
    def forward(self, *x): return torch.cat(*x, dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'


class Unfold(Module):
    def __init__(self, dim, size, step=1): self.dim, self.size, self.step =  dim, size, step
    def forward(self, x:Tensor) -> Tensor: return x.unfold(dimension=self.dim, size=self.size, step=self.step)
    def __repr__(self): return f"{self.__class__.__name__}(dim={self.dim}, size={self.size}, step={self.step})"
    
    
class Permute(Module):
    def __init__(self, *dims): self.dims = dims
    def forward(self, x:Tensor) -> Tensor: return x.permute(self.dims)
    def __repr__(self): return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])})"
    
    
class Transpose(Module):
    def __init__(self, *dims, contiguous=False): self.dims, self.contiguous = dims, contiguous
    def forward(self, x): 
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
    def __repr__(self): 
        if self.contiguous: return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])}).contiguous()"
        else: return f"{self.__class__.__name__}({', '.join([str(d) for d in self.dims])})"
    
    
class View(Module):
    def __init__(self, *shape): self.shape = shape
    def forward(self, x): 
        return x.view(x.shape[0], -1).contiguous() if not self.shape else x.view(-1).contiguous() if self.shape == (-1,) else \
            x.view(x.shape[0], *self.shape).contiguous()
    def __repr__(self): return f"{self.__class__.__name__}({', '.join(['bs'] + [str(s) for s in self.shape])})"
    
    
class Reshape(Module):
    def __init__(self, *shape): self.shape = shape
    def forward(self, x):
        return x.reshape(x.shape[0], -1) if not self.shape else x.reshape(-1) if self.shape == (-1,) else x.reshape(x.shape[0], *self.shape)
    def __repr__(self): return f"{self.__class__.__name__}({', '.join(['bs'] + [str(s) for s in self.shape])})"
    
    
class Max(Module):
    def __init__(self, dim=None, keepdim=False): self.dim, self.keepdim = dim, keepdim
    def forward(self, x): return x.max(self.dim, keepdim=self.keepdim)[0]
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim}, keepdim={self.keepdim})'

    
class LastStep(Module):
    def forward(self, x): return x[..., -1]
    def __repr__(self): return f'{self.__class__.__name__}()'
    
    
class SoftMax(Module):
    "SoftMax layer"
    def __init__(self, dim=-1):
        self.dim = dim
    def forward(self, x):
        return F.softmax(x, dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})' 
    

class Clamp(Module):
    def __init__(self, min=None, max=None):
        self.min, self.max = min, max
    def forward(self, x):
        return x.clamp(min=self.min, max=self.max)
    def __repr__(self): return f'{self.__class__.__name__}(min={self.min}, max={self.max})' 
    
    
class Clip(Module):
    def __init__(self, min=None, max=None):
        self.min, self.max = min, max
        
    def forward(self, x):
        if self.min is not None:
            x = torch.maximum(x, self.min)
        if self.max is not None:
            x = torch.minimum(x, self.max)
        return x
    def __repr__(self): return f'{self.__class__.__name__}()' 
    
    
class ReZero(Module):
    def __init__(self, module):
        self.module = module
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return x + self.alpha * self.module(x)
    
    
Noop = nn.Sequential()

# %% ../../nbs/029_models.layers.ipynb 38
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    It's similar to Dropout but it drops individual connections instead of nodes.
    Original code in https://github.com/rwightman/pytorch-image-models (timm library)
    """

    def __init__(self, p=None):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0. or not self.training: return x
        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
#         output = x.div(random_tensor.mean()) * random_tensor # divide by the actual mean to mantain the input mean?
        return output

# %% ../../nbs/029_models.layers.ipynb 40
class Sharpen(Module):
    "This is used to increase confidence in predictions - MixMatch paper"
    def __init__(self, T=.5): self.T = T
    def forward(self, x):
        x = x**(1. / self.T)
        return x / x.sum(dim=1, keepdims=True)

# %% ../../nbs/029_models.layers.ipynb 42
class Sequential(nn.Sequential):
    """Class that allows you to pass one or multiple inputs"""
    def forward(self, *x):
        for i, module in enumerate(self._modules.values()): 
            x = module(*x) if isinstance(x, (list, tuple, L)) else module(x)
        return x

# %% ../../nbs/029_models.layers.ipynb 43
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

# %% ../../nbs/029_models.layers.ipynb 44
class Temp_Scale(Module):
    "Used to perform Temperature Scaling (dirichlet=False) or Single-parameter Dirichlet calibration (dirichlet=True)"
    def __init__(self, temp=1., dirichlet=False):
        self.weight = nn.Parameter(tensor(temp))
        self.bias = None
        self.log_softmax = dirichlet

    def forward(self, x):
        if self.log_softmax: x = F.log_softmax(x, dim=-1)
        return x.div(self.weight)


class Vector_Scale(Module):
    "Used to perform Vector Scaling (dirichlet=False) or Diagonal Dirichlet calibration (dirichlet=True)"
    def __init__(self, n_classes=1, dirichlet=False):
        self.weight = nn.Parameter(torch.ones(n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.log_softmax = dirichlet

    def forward(self, x):
        if self.log_softmax: x = F.log_softmax(x, dim=-1)
        return x.mul(self.weight).add(self.bias)


class Matrix_Scale(Module):
    "Used to perform Matrix Scaling (dirichlet=False) or Dirichlet calibration (dirichlet=True)"
    def __init__(self, n_classes=1, dirichlet=False):
        self.ms = nn.Linear(n_classes, n_classes)
        self.ms.weight.data = nn.Parameter(torch.eye(n_classes))
        nn.init.constant_(self.ms.bias.data, 0.)
        self.weight = self.ms.weight
        self.bias = self.ms.bias
        self.log_softmax = dirichlet

    def forward(self, x):
        if self.log_softmax: x = F.log_softmax(x, dim=-1)
        return self.ms(x)
    
    
def get_calibrator(calibrator=None, n_classes=1, **kwargs):
    if calibrator is None or not calibrator: return noop
    elif calibrator.lower() == 'temp': return Temp_Scale(dirichlet=False, **kwargs)
    elif calibrator.lower() == 'vector': return Vector_Scale(n_classes=n_classes, dirichlet=False, **kwargs)
    elif calibrator.lower() == 'matrix': return Matrix_Scale(n_classes=n_classes, dirichlet=False, **kwargs)
    elif calibrator.lower() == 'dtemp': return Temp_Scale(dirichlet=True, **kwargs)
    elif calibrator.lower() == 'dvector': return Vector_Scale(n_classes=n_classes, dirichlet=True, **kwargs)
    elif calibrator.lower() == 'dmatrix': return Matrix_Scale(n_classes=n_classes, dirichlet=True, **kwargs)
    else: assert False, f'please, select a correct calibrator instead of {calibrator}'

# %% ../../nbs/029_models.layers.ipynb 49
class LogitAdjustmentLayer(Module):
    "Logit Adjustment for imbalanced datasets"
    def __init__(self, class_priors):
        self.class_priors = class_priors
    def forward(self, x):
        return x.add(self.class_priors)
    
LogitAdjLayer = LogitAdjustmentLayer

# %% ../../nbs/029_models.layers.ipynb 51
class PPV(Module):
    def __init__(self, dim=-1): 
        self.dim = dim
    def forward(self, x): 
        return torch.gt(x, 0).sum(dim=self.dim).float() / x.shape[self.dim]
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'
    

class PPAuc(Module):
    def __init__(self, dim=-1): 
        self.dim = dim
    def forward(self, x): 
        x = F.relu(x).sum(self.dim) / (abs(x).sum(self.dim) + 1e-8)
        return x
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'
    
    
class MaxPPVPool1d(Module):
    "Drop-in replacement for AdaptiveConcatPool1d - multiplies nf by 2"
    def forward(self, x):
        _max = x.max(dim=-1).values
        _ppv = torch.gt(x, 0).sum(dim=-1).float() / x.shape[-1]
        return torch.cat((_max, _ppv), dim=-1).unsqueeze(2)

# %% ../../nbs/029_models.layers.ipynb 53
class AdaptiveWeightedAvgPool1d(Module):
    '''Global Pooling layer that performs a weighted average along the temporal axis
    
    It can be considered as a channel-wise form of local temporal attention. Inspired by the paper: 
    Hyun, J., Seong, H., & Kim, E. (2019). Universal Pooling--A New Pooling Method for Convolutional Neural Networks. arXiv preprint arXiv:1907.11440.'''

    def __init__(self, n_in, seq_len, mult=2, n_layers=2, ln=False, dropout=0.5, act=nn.ReLU(), zero_init=True):
        layers = nn.ModuleList()
        for i in range(n_layers):
            inp_mult = mult if i > 0 else 1
            out_mult = mult if i < n_layers -1 else 1
            p = dropout[i] if is_listy(dropout) else dropout
            layers.append(LinLnDrop(seq_len * inp_mult, seq_len * out_mult, ln=False, p=p, 
                                    act=act if i < n_layers-1 and n_layers > 1 else None))
        self.layers = layers
        self.softmax = SoftMax(-1)
        if zero_init: init_lin_zero(self)

    def forward(self, x):
        wap = x
        for l in self.layers: wap = l(wap)
        wap = self.softmax(wap)
        return torch.mul(x, wap).sum(-1)

# %% ../../nbs/029_models.layers.ipynb 54
class GAP1d(Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Reshape()
    def forward(self, x):
        return self.flatten(self.gap(x))
    
    
class GACP1d(Module):
    "Global AdaptiveConcatPool + Flatten"
    def __init__(self, output_size=1):
        self.gacp = AdaptiveConcatPool1d(output_size)
        self.flatten = Reshape()
    def forward(self, x):
        return self.flatten(self.gacp(x))
    

class GAWP1d(Module):
    "Global AdaptiveWeightedAvgPool1d + Flatten"
    def __init__(self, n_in, seq_len, n_layers=2, ln=False, dropout=0.5, act=nn.ReLU(), zero_init=False):
        self.gacp = AdaptiveWeightedAvgPool1d(n_in, seq_len, n_layers=n_layers, ln=ln, dropout=dropout, act=act, zero_init=zero_init)
        self.flatten = Reshape()
    def forward(self, x):
        return self.flatten(self.gacp(x))

# %% ../../nbs/029_models.layers.ipynb 55
class GlobalWeightedAveragePool1d(Module):
    """ Global Weighted Average Pooling layer 
    
    Inspired by Building Efficient CNN Architecture for Offline Handwritten Chinese Character Recognition
    https://arxiv.org/pdf/1804.01259.pdf
    """
    
    def __init__(self, n_in, seq_len): 
        self.weight = nn.Parameter(torch.ones(1, n_in, seq_len))
        self.bias = nn.Parameter(torch.zeros(1, n_in, seq_len))

    def forward(self, x):
        α = F.softmax(torch.sigmoid(x * self.weight + self.bias), dim=-1)
        return (x * α).sum(-1)

GWAP1d = GlobalWeightedAveragePool1d

def gwa_pool_head(n_in, c_out, seq_len, bn=True, fc_dropout=0.):
    return nn.Sequential(GlobalWeightedAveragePool1d(n_in, seq_len), Reshape(), LinBnDrop(n_in, c_out, p=fc_dropout, bn=bn))

# %% ../../nbs/029_models.layers.ipynb 57
class AttentionalPool1d(Module):
    """Global Adaptive Pooling layer inspired by Attentional Pooling for Action Recognition https://arxiv.org/abs/1711.01467"""
    def __init__(self, n_in, c_out, bn=False): 
        store_attr()
        self.bn = nn.BatchNorm1d(n_in) if bn else None
        self.conv1 = Conv1d(n_in, 1, 1)
        self.conv2 = Conv1d(n_in, c_out, 1)

    def forward(self, x):
        if self.bn is not None: x = self.bn(x) 
        return (self.conv1(x) @ self.conv2(x).transpose(1,2)).transpose(1,2)
    
class GAttP1d(nn.Sequential):
    def __init__(self, n_in, c_out, bn=False):
        super().__init__(AttentionalPool1d(n_in, c_out, bn=bn), Reshape())
        
def attentional_pool_head(n_in, c_out, seq_len=None, bn=True, **kwargs):
    return nn.Sequential(AttentionalPool1d(n_in, c_out, bn=bn, **kwargs), Reshape())

# %% ../../nbs/029_models.layers.ipynb 60
class PoolingLayer(Module):
    def __init__(self, method='cls', seq_len=None, token=True, seq_last=True): 
        method = method.lower()
        assert method in ['cls', 'max', 'mean', 'max-mean', 'linear', 'conv1d', 'flatten']
        if method == 'cls': assert token, 'you can only choose method=cls if a token exists'
        self.method = method
        self.token = token
        self.seq_last = seq_last
        if method == 'linear' or method == 'conv1d':
            self.linear = nn.Linear(seq_len - token, 1)

    def forward(self, x): 
        if self.method == 'cls':
            return x[..., 0] if self.seq_last else x[:, 0]
        if self.token:
            x = x[..., 1:] if self.seq_last else x[:, 1:] 
        if self.method == 'max':
            return torch.max(x, -1)[0] if self.seq_last else torch.max(x, 1)[0]
        elif self.method == 'mean':
            return torch.mean(x, -1) if self.seq_last else torch.mean(x, 1)
        elif self.method == 'max-mean':
            return torch.cat([torch.max(x, -1)[0] if self.seq_last else torch.max(x, 1)[0],
                              torch.mean(x, -1) if self.seq_last else torch.mean(x, 1)], 1)
        elif self.method == 'flatten':
            return x.flatten(1)
        elif self.method == 'linear' or self.method == 'conv1d':
            return self.linear(x)[...,0] if self.seq_last else self.linear(x.transpose(1,2))[...,0]
    
    def __repr__(self): return f"{self.__class__.__name__}(method={self.method}, token={self.token}, seq_last={self.seq_last})"

# %% ../../nbs/029_models.layers.ipynb 63
class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class ReGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.relu(gates)

# %% ../../nbs/029_models.layers.ipynb 64
pytorch_acts = [nn.ELU, nn.LeakyReLU, nn.PReLU, nn.ReLU, nn.ReLU6, nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid, Mish, nn.Softplus,
nn.Tanh, nn.Softmax, GEGLU, ReGLU, SmeLU]
pytorch_act_names = [a.__name__.lower() for a in pytorch_acts]

def get_act_fn(act, **act_kwargs):
    if act is None: return
    elif isinstance(act, nn.Module): return act
    elif callable(act): return act(**act_kwargs)
    idx = pytorch_act_names.index(act.lower())
    return pytorch_acts[idx](**act_kwargs)

# %% ../../nbs/029_models.layers.ipynb 66
class RevIN(nn.Module):
    """ Reversible Instance Normalization layer adapted from

        Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2021, September). 
        Reversible instance normalization for accurate time-series forecasting against distribution shift. 
        In International Conference on Learning Representations.
        Original code: https://github.com/ts-kim/RevIN
    """

    def __init__(self,
         c_in:int,    # #features (aka variables or channels)
         affine:bool=True,  # flag to incidate if RevIN has learnable weight and bias
         subtract_last:bool=False,
         dim:int=2,   # int or tuple of dimensions used to calculate mean and std
         eps:float=1e-5  # epsilon - parameter added for numerical stability
         ):
        super().__init__()
        self.c_in, self.affine, self.subtract_last, self.dim, self.eps = c_in, affine, subtract_last, dim, eps
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, c_in, 1))
            self.bias = nn.Parameter(torch.zeros(1, c_in, 1))
    
    def forward(self, x:Tensor, mode:Tensor):
        """Args:

            x: rank 3 tensor with shape [batch size x c_in x sequence length]
            mode: torch.tensor(True) to normalize data and torch.tensor(False) to reverse normalization
        """
        
        # Normalize
        if mode: return self.normalize(x)
        
        # Denormalize
        else: return self.denormalize(x)
           
    def normalize(self, x):
        if self.subtract_last:
            self.sub = x[..., -1].unsqueeze(-1).detach()
        else:
            self.sub = torch.mean(x, dim=-1, keepdim=True).detach()
        self.std = torch.std(x, dim=-1, keepdim=True, unbiased=False).detach() + self.eps
        if self.affine:
            x = x.sub(self.sub)
            x = x.div(self.std)
            x = x.mul(self.weight)
            x = x.add(self.bias)
            return x
        else:
            x = x.sub(self.sub)
            x = x.div(self.std)
            return x
        
    def denormalize(self, x):
        if self.affine:
            x = x.sub(self.bias)
            x = x.div(self.weight)
            x = x.mul(self.std)
            x = x.add(self.sub)
            return x
        else:
            x = x.mul(self.std)
            x = x.add(self.sub)
            return x

# %% ../../nbs/029_models.layers.ipynb 67
class RevIN(nn.Module):
    """ Reversible Instance Normalization layer adapted from

        Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2021, September). 
        Reversible instance normalization for accurate time-series forecasting against distribution shift. 
        In International Conference on Learning Representations.
        Original code: https://github.com/ts-kim/RevIN
    """

    def __init__(self,
         c_in:int,    # #features (aka variables or channels)
         affine:bool=True,  # flag to incidate if RevIN has learnable weight and bias
         subtract_last:bool=False,
         dim:int=2,   # int or tuple of dimensions used to calculate mean and std
         eps:float=1e-5  # epsilon - parameter added for numerical stability
         ):
        super().__init__()
        self.c_in, self.affine, self.subtract_last, self.dim, self.eps = c_in, affine, subtract_last, dim, eps
        self.weight = nn.Parameter(torch.ones(1, c_in, 1))
        self.bias = nn.Parameter(torch.zeros(1, c_in, 1))
        self.sub, self.std, self.mul, self.add = torch.zeros(1), torch.ones(1), torch.ones(1), torch.zeros(1)
    
    def forward(self, x:Tensor, mode:Tensor):
        """Args:

            x: rank 3 tensor with shape [batch size x c_in x sequence length]
            mode: torch.tensor(True) to normalize data and torch.tensor(False) to reverse normalization
        """
        
        # Normalize
        if mode: 
            if self.subtract_last:
                self.sub = x[..., -1].unsqueeze(-1).detach()
            else:
                self.sub = torch.mean(x, dim=-1, keepdim=True).detach()
            self.std = torch.std(x, dim=-1, keepdim=True, unbiased=False).detach() + self.eps
            if self.affine:
                x = x.sub(self.sub)
                x = x.div(self.std)
                x = x.mul(self.weight)
                x = x.add(self.bias)
                return x
            else:
                x = x.sub(self.sub)
                x = x.div(self.std)
                return x
        
        # Denormalize
        else: 
            if self.affine:
                x = x.sub(self.bias)
                x = x.div(self.weight)
                x = x.mul(self.std)
                x = x.add(self.sub)
                return x
            else:
                x = x.mul(self.std)
                x = x.add(self.sub)
                return x

# %% ../../nbs/029_models.layers.ipynb 70
def create_pool_head(n_in, c_out, seq_len=None, concat_pool=False, fc_dropout=0., bn=False, y_range=None, **kwargs):
    if kwargs: print(f'{kwargs}  not being used')
    if concat_pool: n_in*=2
    layers = [GACP1d(1) if concat_pool else GAP1d(1)]
    layers += [LinBnDrop(n_in, c_out, bn=bn, p=fc_dropout)]
    if y_range: layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)

pool_head = create_pool_head
average_pool_head = partial(pool_head, concat_pool=False)
setattr(average_pool_head, "__name__", "average_pool_head")
concat_pool_head = partial(pool_head, concat_pool=True)
setattr(concat_pool_head, "__name__", "concat_pool_head")

# %% ../../nbs/029_models.layers.ipynb 72
def max_pool_head(n_in, c_out, seq_len, fc_dropout=0., bn=False, y_range=None, **kwargs):
    if kwargs: print(f'{kwargs}  not being used')
    layers = [nn.MaxPool1d(seq_len, **kwargs), Reshape()]
    layers += [LinBnDrop(n_in, c_out, bn=bn, p=fc_dropout)]
    if y_range: layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)

# %% ../../nbs/029_models.layers.ipynb 74
def create_pool_plus_head(*args, lin_ftrs=None, fc_dropout=0., concat_pool=True, bn_final=False, lin_first=False, y_range=None):
    nf = args[0]
    c_out = args[1]
    if concat_pool: nf = nf * 2
    lin_ftrs = [nf, 512, c_out] if lin_ftrs is None else [nf] + lin_ftrs + [c_out]
    ps = L(fc_dropout)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool1d() if concat_pool else nn.AdaptiveAvgPool1d(1)
    layers = [pool, Reshape()]
    if lin_first: layers.append(nn.Dropout(ps.pop(0)))
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += LinBnDrop(ni, no, bn=True, p=p, act=actn, lin_first=lin_first)
    if lin_first: layers.append(nn.Linear(lin_ftrs[-2], c_out))
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if y_range is not None: layers.append(SigmoidRange(*y_range))
    return nn.Sequential(*layers)

pool_plus_head = create_pool_plus_head

# %% ../../nbs/029_models.layers.ipynb 76
def create_conv_head(*args, adaptive_size=None, y_range=None):
    nf = args[0]
    c_out = args[1]
    layers = [nn.AdaptiveAvgPool1d(adaptive_size)] if adaptive_size is not None else []
    for i in range(2):
        if nf > 1: 
            layers += [ConvBlock(nf, nf // 2, 1)] 
            nf = nf//2
        else: break
    layers += [ConvBlock(nf, c_out, 1), GAP1d(1)]
    if y_range: layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)

conv_head = create_conv_head

# %% ../../nbs/029_models.layers.ipynb 78
def create_mlp_head(nf, c_out, seq_len=None, flatten=True, fc_dropout=0., bn=False, lin_first=False, y_range=None):
    if flatten: nf *= seq_len
    layers = [Reshape()] if flatten else []
    layers += [LinBnDrop(nf, c_out, bn=bn, p=fc_dropout, lin_first=lin_first)]
    if y_range: layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)

mlp_head = create_mlp_head

# %% ../../nbs/029_models.layers.ipynb 80
def create_fc_head(nf, c_out, seq_len=None, flatten=True, lin_ftrs=None, y_range=None, fc_dropout=0., bn=False, bn_final=False, act=nn.ReLU(inplace=True)):
    if flatten: nf *= seq_len
    layers = [Reshape()] if flatten else []
    lin_ftrs = [nf, 512, c_out] if lin_ftrs is None else [nf] + lin_ftrs + [c_out]
    if not is_listy(fc_dropout): fc_dropout = [fc_dropout]*(len(lin_ftrs) - 1)
    actns = [act for _ in range(len(lin_ftrs) - 2)] + [None]
    layers += [LinBnDrop(lin_ftrs[i], lin_ftrs[i+1], bn=bn and (i!=len(actns)-1 or bn_final), p=p, act=a) for i,(p,a) in enumerate(zip(fc_dropout+[0.], actns))]
    if y_range is not None: layers.append(SigmoidRange(*y_range))
    return nn.Sequential(*layers)

fc_head = create_fc_head

# %% ../../nbs/029_models.layers.ipynb 82
def create_rnn_head(*args, fc_dropout=0., bn=False, y_range=None):
    nf = args[0]
    c_out = args[1]
    layers = [LastStep()]
    layers += [LinBnDrop(nf, c_out, bn=bn, p=fc_dropout)]
    if y_range: layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)

rnn_head = create_rnn_head

# %% ../../nbs/029_models.layers.ipynb 84
def imputation_head(c_in, c_out, seq_len=None, ks=1, y_range=None, fc_dropout=0.):
    layers = [nn.Dropout(fc_dropout), nn.Conv1d(c_in, c_out, ks)]
    if y_range is not None: 
        y_range = (tensor(y_range[0]), tensor(y_range[1]))
        layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)

# %% ../../nbs/029_models.layers.ipynb 86
class create_conv_lin_nd_head(nn.Sequential):
    "Module to create a nd output head"

    def __init__(self, n_in, n_out, seq_len, d, conv_first=True, conv_bn=False, lin_bn=False, fc_dropout=0., **kwargs):

        assert d, "you cannot use an nd head when d is None or 0"
        if is_listy(d):
            fd = 1
            shape = []
            for _d in d:
                fd *= _d
                shape.append(_d)
            if n_out > 1: shape.append(n_out)
        else: 
            fd = d
            shape = [d, n_out] if n_out > 1 else [d]
        
        conv = [BatchNorm(n_in, ndim=1)] if conv_bn else []
        conv.append(Conv1d(n_in, n_out, 1, padding=0, bias=not conv_bn, **kwargs))
        l = [Transpose(-1, -2), BatchNorm(seq_len, ndim=1), Transpose(-1, -2)] if lin_bn else []
        if fc_dropout != 0: l.append(nn.Dropout(fc_dropout))
        lin = [nn.Linear(seq_len, fd, bias=not lin_bn)]
        lin_layers = l+lin
        layers = conv + lin_layers if conv_first else lin_layers + conv
        layers += [Transpose(-1,-2)]
        layers += [Reshape(*shape)]

        super().__init__(*layers)
        
conv_lin_nd_head = create_conv_lin_nd_head
conv_lin_3d_head = create_conv_lin_nd_head # included for compatibility
create_conv_lin_3d_head = create_conv_lin_nd_head # included for compatibility

# %% ../../nbs/029_models.layers.ipynb 91
class lin_nd_head(nn.Sequential):
    "Module to create a nd output head with linear layers"

    def __init__(self, n_in, n_out, seq_len=None, d=None, flatten=False, use_bn=False, fc_dropout=0.):

        if seq_len is None:
            seq_len = 1
        if d is None:
            fd = 1
            shape = [n_out]
        elif is_listy(d):
            fd = 1
            shape = []
            for _d in d:
                fd *= _d
                shape.append(_d)
            if n_out > 1: shape.append(n_out)
        else: 
            fd = d
            shape = [d, n_out] if n_out > 1 else [d]
            
        layers = []
        if use_bn:
            layers += [nn.BatchNorm1d(n_in)]
        if fc_dropout:
            layers += [nn.Dropout(fc_dropout)]
        if d is None:
            if not flatten or seq_len == 1:
                layers += [nn.AdaptiveAvgPool1d(1), Squeeze(-1), nn.Linear(nf, n_out)]
                if n_out == 1:
                    layers += [Squeeze(-1)]
            else:
                layers += [Reshape(), nn.Linear(n_in * seq_len, n_out * fd)]
                if n_out * fd== 1:
                    layers += [Squeeze(-1)]
        else:
            if seq_len == 1:
                layers += [nn.AdaptiveAvgPool1d(1)]
            if not flatten and fd == seq_len:
                layers += [Transpose(1,2), nn.Linear(n_in, n_out)]
            else:
                layers += [Reshape(), nn.Linear(n_in * seq_len, n_out * fd)]
            layers += [Reshape(*shape)]

        super().__init__(*layers)
        
create_lin_nd_head = lin_nd_head
lin_3d_head = lin_nd_head # included for backwards compatiblity
create_lin_3d_head = lin_nd_head # included for backwards compatiblity

# %% ../../nbs/029_models.layers.ipynb 97
class rocket_nd_head(nn.Sequential):
    "Module to create a nd output head with linear layers for the rocket family of models"

    def __init__(self, n_in, n_out, seq_len=None, d=None, use_bn=False, fc_dropout=0., zero_init=True):

        if d is None:
            fd = 1
            shape = [n_out]
        elif is_listy(d):
            fd = 1
            shape = []
            for _d in d:
                fd *= _d
                shape.append(_d)
            if n_out > 1: shape.append(n_out)
        else: 
            fd = d
            shape = [d, n_out] if n_out > 1 else [d]

        layers = [nn.Flatten()]
        if use_bn:
            layers += [nn.BatchNorm1d(n_in)]
        if fc_dropout:
            layers += [nn.Dropout(fc_dropout)]
        linear = nn.Linear(n_in, fd * n_out)
        if zero_init:
            nn.init.constant_(linear.weight.data, 0)
            nn.init.constant_(linear.bias.data, 0)
        layers += [linear]
        if d is None and n_out == 1:
            layers += [Squeeze(-1)]
        if d is not None:
            layers += [Reshape(*shape)]

        super().__init__(*layers)

# %% ../../nbs/029_models.layers.ipynb 99
class xresnet1d_nd_head(nn.Sequential):
    "Module to create a nd output head with linear layers for the xresnet family of models"

    def __init__(self, n_in, n_out, seq_len=None, d=None, use_bn=False, fc_dropout=0., zero_init=True):

        if d is None:
            fd = 1
            shape = [n_out]
        elif is_listy(d):
            fd = 1
            shape = []
            for _d in d:
                fd *= _d
                shape.append(_d)
            if n_out > 1: shape.append(n_out)
        else: 
            fd = d
            shape = [d, n_out] if n_out > 1 else [d]

        layers = [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        if use_bn:
            layers += [nn.BatchNorm1d(n_in)]
        if fc_dropout:
            layers += [nn.Dropout(fc_dropout)]
        linear = nn.Linear(n_in, fd * n_out)
        if zero_init:
            nn.init.constant_(linear.weight.data, 0)
            nn.init.constant_(linear.bias.data, 0)
        layers += [linear]
        if d is None and n_out == 1:
            layers += [Squeeze(-1)]
        if d is not None:
            layers += [Reshape(*shape)]

        super().__init__(*layers)

# %% ../../nbs/029_models.layers.ipynb 101
class create_conv_3d_head(nn.Sequential):
    "Module to create a nd output head with a convolutional layer"
    def __init__(self, n_in, n_out, seq_len, d, use_bn=False, **kwargs):
        assert d, "you cannot use an 3d head when d is None or 0"
        assert d == seq_len, 'You can only use this head when learn.dls.len == learn.dls.d'
        layers = [nn.BatchNorm1d(n_in)] if use_bn else []
        layers += [Conv(n_in, n_out, 1, **kwargs), Transpose(-1,-2)]
        if n_out == 1: layers += [Squeeze(-1)]
        super().__init__(*layers)
        
conv_3d_head = create_conv_3d_head

# %% ../../nbs/029_models.layers.ipynb 104
def universal_pool_head(n_in, c_out, seq_len, mult=2, pool_n_layers=2, pool_ln=True, pool_dropout=0.5, pool_act=nn.ReLU(),
                        zero_init=True, bn=True, fc_dropout=0.):
    return nn.Sequential(AdaptiveWeightedAvgPool1d(n_in, seq_len, n_layers=pool_n_layers, mult=mult, ln=pool_ln, dropout=pool_dropout, act=pool_act), 
                         Reshape(), LinBnDrop(n_in, c_out, p=fc_dropout, bn=bn))

# %% ../../nbs/029_models.layers.ipynb 106
heads = [mlp_head, fc_head, average_pool_head, max_pool_head, concat_pool_head, pool_plus_head, conv_head, rnn_head, 
         conv_lin_nd_head, lin_nd_head, conv_3d_head, attentional_pool_head, universal_pool_head, gwa_pool_head]

# %% ../../nbs/029_models.layers.ipynb 108
class SqueezeExciteBlock(Module):
    def __init__(self, ni, reduction=16):
        self.avg_pool = GAP1d(1)
        self.fc = nn.Sequential(nn.Linear(ni, ni // reduction, bias=False), nn.ReLU(),  nn.Linear(ni // reduction, ni, bias=False), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y).unsqueeze(2)
        return x * y.expand_as(x)

# %% ../../nbs/029_models.layers.ipynb 110
class GaussianNoise(Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        self.sigma, self.is_relative_detach = sigma, is_relative_detach

    def forward(self, x):
        if self.training and self.sigma not in [0, None]:
            scale = self.sigma * (x.detach() if self.is_relative_detach else x)
            sampled_noise = torch.empty(x.size(), device=x.device).normal_() * scale
            x = x + sampled_noise
        return x 

# %% ../../nbs/029_models.layers.ipynb 114
class PositionwiseFeedForward(nn.Sequential):
    def __init__(self, dim, dropout=0., act='reglu', mlp_ratio=1):
        act_mult = 2 if act.lower() in ["geglu", "reglu"] else 1
        super().__init__(nn.Linear(dim, dim * mlp_ratio * act_mult),
                         get_act_fn(act),
                         nn.Dropout(dropout),
                         nn.Linear(dim * mlp_ratio, dim),
                         nn.Dropout(dropout))

class TokenLayer(Module):
    def __init__(self, token=True): self.token = token
    def forward(self, x): return x[..., 0] if self.token is not None else x.mean(-1)
    def __repr__(self): return f"{self.__class__.__name__}()"

# %% ../../nbs/029_models.layers.ipynb 116
class ScaledDotProductAttention(Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer 
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets 
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):  
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]

        Output shape: 
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev 

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

# %% ../../nbs/029_models.layers.ipynb 118
class MultiheadAttention(Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer

        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """

        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights 

# %% ../../nbs/029_models.layers.ipynb 125
class MultiConv1d(Module):
    """Module that applies multiple convolutions with different kernel sizes"""

    def __init__(self, ni, nf=None, kss=[1,3,5,7], keep_original=False, separable=False, dim=1, **kwargs):
        kss = listify(kss)
        n_layers = len(kss)
        if ni == nf: keep_original = False
        if nf is None: nf = ni * (keep_original + n_layers)
        nfs = [(nf - ni*keep_original) // n_layers] * n_layers
        while np.sum(nfs) + ni * keep_original < nf:
            for i in range(len(nfs)):
                nfs[i] += 1
                if np.sum(nfs) + ni * keep_original == nf: break

        _conv = SeparableConv1d if separable else Conv1d
        self.layers = nn.ModuleList()
        for nfi,ksi in zip(nfs, kss):
            self.layers.append(_conv(ni, nfi, ksi, **kwargs))
        self.keep_original, self.dim = keep_original, dim

    def forward(self, x):
        output = [x] if self.keep_original else []
        for l in self.layers:
            output.append(l(x))
        x = torch.cat(output, dim=self.dim)
        return x

# %% ../../nbs/029_models.layers.ipynb 127
class LSTMOutput(Module):
    def forward(self, x): return x[0]
    def __repr__(self): return f'{self.__class__.__name__}()'

# %% ../../nbs/029_models.layers.ipynb 129
def emb_sz_rule(n_cat):
    "Rule of thumb to pick embedding size corresponding to `n_cat` (original from fastai)"
    return min(600, round(1.6 * n_cat**0.56))

# %% ../../nbs/029_models.layers.ipynb 131
class TSEmbedding(nn.Embedding):
    "Embedding layer with truncated normal initialization adapted from fastai"
    def __init__(self, ni, nf, std=0.01, padding_idx=None):
        super().__init__(ni, nf)
        trunc_normal_(self.weight.data, std=std)
        if padding_idx is not None:
            nn.init.zeros_(self.weight.data[padding_idx])

# %% ../../nbs/029_models.layers.ipynb 132
class MultiEmbedding(Module):
    def __init__(self, c_in, n_cat_embeds, cat_embed_dims=None, cat_pos=None, std=0.01, cat_padding_idxs=None):
        cat_n_embeds = listify(n_cat_embeds)
        if cat_padding_idxs is None: cat_padding_idxs = [None]
        else: cat_padding_idxs = listify(cat_padding_idxs)
        if len(cat_padding_idxs) == 1 and len(cat_padding_idxs) < len(cat_n_embeds): 
            cat_padding_idxs = cat_padding_idxs * len(cat_n_embeds)
        assert len(cat_n_embeds) == len(cat_padding_idxs)
        if cat_embed_dims is None: 
            cat_embed_dims = [emb_sz_rule(s) for s in cat_n_embeds]
        else:
            cat_embed_dims = listify(cat_embed_dims)
            if len(cat_embed_dims) == 1: cat_embed_dims = cat_embed_dims * len(cat_n_embeds)
            assert len(cat_embed_dims) == len(cat_n_embeds)
        if cat_pos: 
            cat_pos = torch.as_tensor(listify(cat_pos))  
        else: 
            cat_pos = torch.arange(len(cat_n_embeds))
        self.register_buffer("cat_pos", cat_pos)
        cont_pos = torch.tensor([p for p in torch.arange(c_in) if p not in self.cat_pos])
        self.register_buffer("cont_pos", cont_pos)
        self.cat_embed = nn.ModuleList([TSEmbedding(n,d,std=std, padding_idx=p) for n,d,p in zip(cat_n_embeds, cat_embed_dims, cat_padding_idxs)])

    def forward(self, x):
        if isinstance(x, tuple): x_cat, x_cont, *_ = x
        else: x_cat, x_cont = x[:, self.cat_pos], x[:, self.cont_pos]
        x_cat = torch.cat([e(torch.round(x_cat[:,i]).long()).transpose(1,2) for i,e in enumerate(self.cat_embed)],1)
        return torch.cat([x_cat, x_cont], 1)
