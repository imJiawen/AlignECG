import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm
import time 
import argparse
import sys

from baselines.RNN_FCN import GRU_FCN
from baselines.TransformerModel import TransformerModel
from baselines.FCN import FCN
from baselines.ResNet import ResNet
from baselines.ResCNN import ResCNN
from baselines.RNNAttention import GRUAttention
from baselines.TCN import TCN

from DiffWarping import Almtx
from alignment_loss import alignment_loss

import torch
import torch.nn as nn
import main_utils


def train_epoch(train_dataloader, get_almtx, classifier, criterion, optimizer, device):
    losses = []
    sup_preds = []
    sup_labels = []
    classifier.train()
    for train_batch, label in tqdm(train_dataloader, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        train_batch, label = train_batch.to(device), label.to(device)
        # train_batch: [b 1 l]
        optimizer.zero_grad()
        align_loss = 0

        if get_almtx is not None:
            non_pad_mask = torch.zeros(train_batch.shape).to(device)
            non_pad_mask[train_batch > 0] = 1
            train_batch, thetas = get_almtx(train_batch, mask=non_pad_mask) # [b l s]

            align_loss = alignment_loss(train_batch, label, n_channels=1)


        pred = classifier(train_batch)
        pred_score = torch.softmax(pred, dim=-1)

        loss = criterion(pred, label)

        if torch.any(torch.isnan(loss)):
            print("exit nan in pred loss!!! \n sup_pred \n", pred)
            sys.exit(0)

        loss = torch.sum(loss)
        loss = loss + align_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        sup_preds.append(pred_score.detach().cpu().numpy())
        sup_labels.append(label.detach().cpu().numpy())

    if len(sup_preds) > 0:
        sup_labels = np.concatenate(sup_labels, axis=0)
        sup_preds = np.concatenate(sup_preds, axis=0)
        sup_preds = np.nan_to_num(sup_preds)

    train_loss = np.average(losses)
    auroc, auprc, acc, f1 = main_utils.cal_scores(sup_labels, sup_preds)
    
    return auroc, auprc, acc, f1, train_loss

def eval_epoch(test_dataloader, get_almtx, classifier, criterion, device):
    # Testing
    val_preds = []
    val_labels = []
    val_losses = []
    classifier.eval()
    with torch.no_grad():
        for test_batch, label in tqdm(test_dataloader, mininterval=2,
                            desc='  - (Validation)   ', leave=False):
            test_batch, label = test_batch.to(device), label.to(device)

            if get_almtx is not None:
                # test_batch, thetas = get_almtx(test_batch, return_theta=True)

                non_pad_mask = torch.zeros(test_batch.shape).to(device)
                non_pad_mask[test_batch > 0] = 1
                test_batch, thetas = get_almtx(test_batch, mask=non_pad_mask) # [b l s]
                # test_batch = torch.matmul(test_batch, almat) # [b 1 s]

            pred = classifier(test_batch)
            val_loss = criterion(pred, label)
            pred_score = torch.softmax(pred, dim=-1)
            val_loss = torch.sum(val_loss)
            val_losses.append(val_loss.item())

            val_preds.append(pred_score.detach().cpu().numpy())
            val_labels.append(label.detach().cpu().numpy())


        if len(val_preds) > 0:
            val_labels = np.concatenate(val_labels, axis=0)
            val_preds = np.concatenate(val_preds, axis=0)
            val_preds = np.nan_to_num(val_preds)

        val_loss = np.average(val_losses)
        auroc, auprc, acc, f1 = main_utils.cal_scores(val_labels, val_preds)
        
    return auroc, auprc, acc, f1, val_labels, val_preds, val_loss


def inference(test_dataloader, classifier, save_path, device, get_almtx=None):
    test_preds = []
    test_scores = []
    start = time.time()
    classifier.eval()
    with torch.no_grad():
        for test_batch in tqdm(test_dataloader, mininterval=2,
                            desc='  - (Testing)   ', leave=False):
            # test_batch = test_batch.to(device)
            test_batch = test_batch.to(device)

            if get_almtx is not None:
                # test_batch, thetas = get_almtx(test_batch, return_theta=True)

                non_pad_mask = torch.zeros(test_batch.shape).to(device)
                non_pad_mask[test_batch > 0] = 1
                test_batch, thetas = get_almtx(test_batch, mask=non_pad_mask) # [b l s]


            pred = classifier(test_batch)
            pred_score = torch.softmax(pred, dim=-1)
            pred_label = np.argmax(pred_score.cpu().numpy(), axis=-1)

            test_preds.append(pred_label)


    test_preds = np.concatenate(test_preds, axis=0)
    main_utils.save_res(test_preds, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--log_path', type=str, default='./log/')
    parser.add_argument('--save_path', type=str, default='./res/')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/')
    parser.add_argument('--model', type=str, default='grufcn')

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_cells', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--warptype', type=str, default='cpab')
    parser.add_argument('--align', type=int, default=None)
    args = parser.parse_args()

    n_vars = 1
    n_class = 5
    seq_len = 187

    epoch_num = args.epoch
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_path, save_path, ckpt_path = main_utils.get_names(args)

    esrly_stop = main_utils.EarlyStopping(patience=args.patience, save_path=ckpt_path, verbose=True)

    print('[Info] parameters: {}\n'.format(args)    )

    if not args.inference:
        with open(log_path, 'w') as f:
            f.write('[Info] parameters: {}\n'.format(args))

    # change the seeds in different runs
    main_utils.setup_seed(args.seed)

    train_dataloader, val_dataloader, test_dataloader = main_utils.get_ecg_data(args.batch_size, data_path=args.data_path, inference=args.inference)


    # model definition
    if args.model == 'grufcn':
        classifier = GRU_FCN(n_vars, n_class, hidden_size=16, rnn_layers=2, bidirectional=True, shuffle=False).to(device) # [b 1 l] => [b n]
    elif args.model == 'trans':
        classifier = TransformerModel(n_vars, n_class,n_head=4, n_layers=3).to(device)
    elif args.model == 'fcn':
        classifier = FCN(n_vars, n_class).to(device)
    elif args.model == 'resnet':
        classifier = ResNet(n_vars, n_class).to(device)
    elif args.model == 'rescnn':
        classifier = ResCNN(n_vars, n_class).to(device)
    elif args.model == 'gruatt':
        if args.align is not None:
            seq_len = args.align
        classifier = GRUAttention(c_in=n_vars, c_out=n_class, seq_len=seq_len).to(device)
    elif args.model == 'tcn':
        # batch 512, lr 1e-3
        classifier = TCN(c_in=n_vars, c_out=n_class).to(device)

    else:
        print('invalid model name')
        sys.exit(0)

    if args.align is not None:
        get_almtx_net = Almtx(seq_len, n_vars, args.align, device,  warp=args.warptype)
    else:
        get_almtx_net = None

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(epoch_num):
        start = time.time()
        train_auroc, train_auprc, train_acc, train_f1, train_loss = train_epoch(train_dataloader, get_almtx_net, classifier, criterion, optimizer, device)
        main_utils.log_info('Train', epoch, train_acc, start, auroc=train_auroc, auprc=train_auprc, f1=train_f1, loss=train_loss)

        if val_dataloader is not None:
            start = time.time()
            val_auroc, val_auprc, val_acc, val_f1, _, _, val_loss = eval_epoch(val_dataloader, get_almtx_net, classifier, criterion, device)
            esrly_stop(epoch, val_loss, classifier, get_almtx_net)
            main_utils.log_info('Validation', epoch, val_acc, start, auroc=val_auroc, auprc=val_auprc, f1=val_f1, loss=val_loss, log_path=log_path)
            
            if esrly_stop.early_stop:
                print("Early stopping...")
                classifier, get_almtx_net = main_utils.load_checkpoints(ckpt_path, classifier, get_almtx_net)
                break

        scheduler.step()


    if args.inference:
        inference(test_dataloader, classifier, save_path, device, get_almtx=get_almtx_net)
    else: 
        test_auroc, test_auprc, test_acc, test_f1, test_labels, test_preds, _ = eval_epoch(test_dataloader, get_almtx_net, classifier, criterion, device)
        main_utils.evaluator(test_labels, test_preds, log_path=log_path)
        main_utils.log_info('Testing', esrly_stop.best_epoch, test_acc, start, auroc=test_auroc, auprc=test_auprc, f1=test_f1, loss=0, log_path=log_path)


if __name__ == '__main__':
    main()
    
