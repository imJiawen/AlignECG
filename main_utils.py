import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, label_binarize, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import classification_report
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Setting Dictionary to define the type of Heartbeat for both datasets
Outcome = {0. : 'Class 0',
               1. : 'Class 1',
               2. : 'Class 2',
               3. : 'Class 3',
               4. : 'Class 4'}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)  # gpu
    

def normalization(X_list, norm='minmax'):
    # [train, val, test]
    if norm == 'minmax':
        #Normalizing the training & test data 
        for i in range(len(X_list)):
            X_list[i] = normalize(X_list[i], axis=0, norm='max')

    elif norm == 'zscore':
        scaler=StandardScaler().fit(X_list[0])
        for i in range(len(X_list)):
            X_list[i] = scaler.transform( X_list[i])

    return X_list


def get_ecg_data(batch_size, data_path='', inference=False):

    test = pd.read_csv(data_path + 'test.csv')
    train = pd.read_csv(data_path + 'train.csv')
    train.rename(columns={'187':"Class"}, inplace=True)

    if not inference:
        X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,1:188], train.iloc[:,-1], test_size=0.2, random_state=42, stratify=train.iloc[:,-1])
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        X_valid, y_valid = np.array(X_valid), np.array(y_valid)
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)

        # X_train, X_valid, X_test = normalization([X_train, X_valid, X_test], norm='minmax')

        val_data_combined = TensorDataset(torch.from_numpy(X_valid).unsqueeze(1).float(),
                                        torch.from_numpy(y_valid).long().squeeze())
        val_dataloader = DataLoader(val_data_combined, batch_size=batch_size, shuffle=False)
        test_data_combined = TensorDataset(torch.from_numpy(X_test).unsqueeze(1).float(),
                                        torch.from_numpy(y_test).long().squeeze())

    else:
        X_train, y_train = np.array(train.iloc[:,1:188]), np.array(train.iloc[:,-1])
        X_test = np.array(test.iloc[:,1:])
        # X_train, X_test = normalization([X_train, X_test], norm='minmax')
        val_dataloader = None
        test_data_combined = torch.from_numpy(X_test).unsqueeze(1).float()

        
    train_data_combined = TensorDataset(torch.from_numpy(X_train).unsqueeze(1).float(),
                                    torch.from_numpy(y_train).long().squeeze())

    train_dataloader = DataLoader(train_data_combined, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data_combined, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def cal_scores(true_score, pred_scores):
    labels_classes = label_binarize(true_score, classes=range(5))
    idx = np.argmax(pred_scores, axis=-1)
    preds_label = np.zeros(pred_scores.shape)
    preds_label[np.arange(preds_label.shape[0]), idx] = 1

    auroc = roc_auc_score(labels_classes, pred_scores, average='macro')
    auprc = average_precision_score(labels_classes, pred_scores, average='macro')
    acc = accuracy_score(labels_classes, preds_label)
    f1 = f1_score(true_score, idx, average='macro')

    return auroc, auprc, acc, f1

def log_info(phase, epoch, acc, start, auroc=0, auprc=0, f1=0, loss=0, log_path=None):
    
    print('  -(', phase, ') epoch: {epoch}, acc: {type: 8.5f}, '
                'AUROC: {auroc: 8.5f}, AUPRC: {auprc: 8.5f}, F1 score: {f1: 8.5f}, loss: {loss: 8.5f}, elapse: {elapse:3.3f} min'
                .format(phase=phase,epoch=epoch,type=acc, auroc=auroc, auprc=auprc, f1=f1, loss=loss, elapse=(time.time() - start) / 60))
    
    if log_path is not None:
        with open(log_path, 'a') as f:
            f.write(str(phase) + ':\t{epoch}, ACC: {acc: 8.5f}, AUROC: {auroc: 8.5f}, AUPRC: {auprc: 8.5f}, F1 score: {f1: 8.5f},  Loss: {loss: 8.5f}\n'
                    .format(epoch=epoch, acc=acc, auroc=auroc, auprc=auprc, f1=f1, loss=loss))


def evaluator(labels, preds, log_path=None):
    auroc, auprc, acc, f1 = cal_scores(labels, preds)
    print("ACC: {}, AUROC: {}, AUPRC: {}, F1 score: {}".format(acc, auroc, auprc, f1))

    preds_label = np.argmax(preds, axis=-1)
    report = classification_report(labels, preds_label.astype(int), target_names=[Outcome[i] for i in Outcome])
    print(report)
    if log_path is not None:
        with open(log_path, 'a') as f:
            f.write(report + '\n')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, save_path=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.best_epoch = 0

    def __call__(self, epoch, val_loss, model, get_almtx_net=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, get_almtx_net)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model, get_almtx_net)
            self.counter = 0


    def save_checkpoint(self, val_loss, model, get_almtx_net=None):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if get_almtx_net is not None:
            get_almtx_net_state = get_almtx_net.state_dict()
        else:
            get_almtx_net_state = None

        torch.save({
            'best_epoch': self.best_epoch,
            'model_state_dict': model.state_dict(),
            'almtx_state_dict': get_almtx_net_state
        }, self.save_path)

        self.val_loss_min = val_loss


def load_checkpoints(save_path, model=None, get_almtx_net=None, use_cpu=False):
    
    if use_cpu:
        checkpoint = torch.load(save_path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(save_path)
    
    if model is not None and checkpoint['model_state_dict'] is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    if get_almtx_net is not None and checkpoint['almtx_state_dict'] is not None:
        get_almtx_net.load_state_dict(checkpoint['almtx_state_dict'])
        
    return model, get_almtx_net

def get_names(args):
    log_path = args.log_path + args.model 
    save_path = args.save_path + args.model 
    ckpt_path = args.ckpt_path + args.model 

    if args.align is not None:
        log_path = log_path + '_' + args.warptype + str(args.align) 
        save_path = save_path + '_' + args.warptype + str(args.align) 
        ckpt_path = ckpt_path + '_' + args.warptype + str(args.align) 

    log_path = log_path + '_seed' + str(args.seed) + '.log'
    save_path = save_path + '_seed' + str(args.seed) + '.csv'
    ckpt_path = ckpt_path + '_seed' + str(args.seed) + '.h5'

    return log_path, save_path, ckpt_path

def save_res(test_preds, save_path):
    id_list = np.arange(1, len(test_preds)+1)
    dataframe = pd.DataFrame({'id': id_list, 'predicted': test_preds})
    dataframe.to_csv(save_path, index=False, sep=',')
    print('save to ', save_path, ' done')