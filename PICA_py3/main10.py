#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA
import pickle
import os
import sys
sys.path.append('..')
import time
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from lib import Config as cfg, networks, datasets, Session
from lib.utils import (lr_policy, optimizers, transforms, save_checkpoint, 
                            AverageMeter, TimeProgressMeter, traverse)
from lib.utils.loggers import STDLogger as logger, TFBLogger as SummaryWriter

from pica.utils import ConcatDataset, RepeatSampler, RandomSampler, get_reduced_transform
from pica.losses import PUILoss
#from lib.datasets.new_dataset import NewDataSet, NewDataSet_test
from lib.datasets.pickle10_dataset import NewDataSet, NewDataSet_test
import torchsummary
def require_args():

    # args for training
    cfg.add_argument('--max-epochs', default=350, type=int,
                        help='maximal training epoch')
    cfg.add_argument('--display-freq', default=80, type=int,
                        help='log display frequency')
    cfg.add_argument('--batch-size', default=256, type=int,
                        help='size of mini-batch')
    cfg.add_argument('--num-workers', default=2, type=int,
                        help='number of workers used for loading data')
    cfg.add_argument('--data-nrepeat', default=2, type=int,
                        help='how many times each image in a ' +
                             'mini-batch should be repeated')
    cfg.add_argument('--pica-lamda', default=2.0, type=float,
                        help='weight of negative entropy regularisation')

def main():

    logger.info('Start to declare training variable')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Session will be ran in device: [%s]' % cfg.device)
    start_epoch = 0
    best_acc = 0.

    logger.info('Start to prepare data')

    # otrainset: original trainset
    otrainset = [NewDataSet()]
    logger.info(f'otrainset----------------------: length {len(otrainset[0])}')

    # ptrainset: perturbed trainset
    ptrainset = [NewDataSet()]
    logger.info(f'ptrainset----------------------: length {len(ptrainset[0])}')

    # testset
    testset = NewDataSet_test()
    logger.info(f'testset-------------: length {len(testset)}')
    # declare data loaders for testset only
    test_loader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, 
                                num_workers=cfg.num_workers)
    logger.info('Start to build model')
    net = networks.get()
    torchsummary.summary(net.cuda(), input_size=(1, 32, 32, 32))
    print(net)
    criterion = PUILoss(cfg.pica_lamda)
    optimizer = optimizers.get(params=[val for _, val in net.trainable_parameters().items()])
    lr_handler = lr_policy.get()

    # load session if checkpoint is provided
    if cfg.resume:
        print("RESUME FOUND!!!!")
        assert os.path.exists(cfg.resume), "Resume file not found"
        ckpt = torch.load(cfg.resume)
        logger.info('Start to resume session for file: [%s]' % cfg.resume)
        net.load_state_dict(ckpt['net'])
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']

    # data parallel
    if cfg.device == 'cuda' and len(cfg.gpus.split(',')) > 1:
        logger.info('Data parallel will be used for acceleration purpose')
        device_ids = range(len(cfg.gpus.split(',')))
        if not (hasattr(net, 'data_parallel') and net.data_parallel(device_ids)):
            net = nn.DataParallel(net, device_ids=device_ids)
        cudnn.benchmark = True
    else:
        logger.info('Data parallel will not be used for acceleration')

    # move modules to target device
    net, criterion = net.to(cfg.device), criterion.to(cfg.device)

    # tensorboard wrtier
    writer = SummaryWriter(cfg.debug, log_dir=cfg.tfb_dir)
    # start training
    lr = cfg.base_lr
    epoch = start_epoch
    while lr > 0 and epoch < cfg.max_epochs:

        lr = lr_handler.update(epoch, optimizer)
        writer.add_scalar('Train/Learing_Rate', lr, epoch)

        logger.info('Start to train at %d epoch with learning rate %.6f' % (epoch, lr))
        train(epoch, net, otrainset, ptrainset, optimizer, criterion, writer)

        logger.info('Start to evaluate after %d epoch of training' % epoch)
        acc, nmi, ari = evaluate(net, test_loader)
        logger.info('Evaluation results at epoch %d are: '
            'ACC: %.3f, NMI: %.3f, ARI: %.3f' % (epoch, acc, nmi, ari))
        writer.add_scalar('Evaluate/ACC', acc, epoch)
        writer.add_scalar('Evaluate/NMI', nmi, epoch)
        writer.add_scalar('Evaluate/ARI', ari, epoch)

        epoch += 1

        if cfg.debug:
            continue

        # save checkpoint
        is_best = acc > best_acc
        best_acc = max(best_acc, acc)
        save_checkpoint({'net' : net.state_dict(), 
                'optimizer' : optimizer.state_dict(),
                'acc' : acc,
                'epoch' : epoch}, is_best=is_best)

    logger.info('Done')

def train(epoch, net, otrainset, ptrainset, optimizer, criterion, writer):
    """alternate the training of different heads
    """
    logger.info('cfg.net_heads: {cfg.net_heads}')
    for hidx, head in enumerate(cfg.net_heads):
        logger.info(f'hidx, head: {hidx}, {head}')
    for hidx, head in enumerate(cfg.net_heads):
        train_head(epoch, net, hidx, head, otrainset[min(len(otrainset) - 1, hidx)], 
            ptrainset[min(len(ptrainset) - 1, hidx)], optimizer, criterion, writer)

def train_head(epoch, net, hidx, head, otrainset, ptrainset, optimizer, criterion, writer):
    """trains one head for an epoch
    """
    logger.info(f'train_head-------------: otrainset={len(otrainset)}, ptrainset={len(ptrainset)}')
    # declare dataloader
    random_sampler = RandomSampler(otrainset)
    batch_sampler = RepeatSampler(random_sampler, cfg.batch_size, nrepeat=cfg.data_nrepeat)
    ploader = DataLoader(ptrainset, batch_sampler=batch_sampler, 
                        num_workers=cfg.num_workers, pin_memory=True)
    oloader = DataLoader(otrainset, sampler=random_sampler, 
                        batch_size=cfg.batch_size, num_workers=cfg.num_workers, 
                        pin_memory=True)
    
    # set network mode
    net.train()

    # tracking variable
    end = time.time()
    train_loss = AverageMeter('Loss', ':.4f')
    data_time = AverageMeter('Data', ':.3f')
    batch_time = AverageMeter('Time', ':.3f')
    progress = TimeProgressMeter(batch_time, data_time, train_loss, 
            Batch=len(oloader), Head=len(cfg.net_heads), Epoch=cfg.max_epochs)

    for batch_idx, (obatch, pbatch) in enumerate(zip(oloader, ploader)):
        # record data loading time
        data_time.update(time.time() - end)

        # move data to target device
        (oinputs, ol), (pinputs, pl) = (obatch, pbatch)
        oinputs, pinputs = (oinputs.to(cfg.device, non_blocking=True), 
                            pinputs.to(cfg.device, non_blocking=True))
        #oinputs = Variable(oinputs, requires_grad=True)
        #pinputs = Variable(pinputs, requires_grad=True)
        #prev=time.time()
        # forward
        ologits, plogits = net(oinputs)[hidx], net(pinputs)[hidx]
        #ologits = Variable(ologits, requires_grad=True)
        #plogits = Variable(plogits, requires_grad=True)
        #print(f'training: {time.time()-prev}')
        #prev=time.time()
        loss = criterion(ologits.repeat(cfg.data_nrepeat, 1), plogits)
        # print(ologits.repeat(cfg.data_nrepeat, 1))
        # print(loss)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(f'loss and optimization: {time.time()-prev}')
        train_loss.update(loss.item(), oinputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar('Train/Loss/Head-%d' % head, train_loss.val, epoch * len(oloader) + batch_idx)

        if batch_idx % cfg.display_freq != 0:
            continue

        logger.info(progress.show(Batch=batch_idx, Epoch=epoch, Head=hidx))

def evaluate(net, loader):
    """evaluates on provided data
    """

    net.eval()
    predicts = np.zeros(len(loader.dataset), dtype=np.int32)
    logger.info('len(loader.dataset)')
    logger.info(len(loader.dataset))
    labels = np.zeros(len(loader.dataset), dtype=np.int32)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if(batch_idx % 10 == 0):
                logger.progress('processing %d/%d batch' % (batch_idx, len(loader)))
            inputs = inputs.to(cfg.device, non_blocking=True)
            # assuming the last head is the main one
            # output dimension of the last head 
            # should be consistent with the ground-truth
            logits = net(inputs)[-1]
            start = batch_idx * loader.batch_size
            end = start + loader.batch_size
            end = min(end, len(loader.dataset))
            #prevTime = time.time()
            labels[start:end] = targets.cpu().numpy()
            predicts[start:end] = logits.max(1)[1].cpu().numpy()


    # compute accuracy
    num_classes = 10#labels.max().item() + 1
    logger.info('num_classes')
    logger.info(num_classes)
    count_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for i in range(predicts.shape[0]):
        count_matrix[predicts[i], labels[i]] += 1
    logger.info(count_matrix)
    reassignment = np.dstack(linear_sum_assignment(count_matrix.max() - count_matrix))[0]
    acc = count_matrix[reassignment[:,0], reassignment[:,1]].sum().astype(np.float32) / predicts.shape[0]
    return acc, NMI(labels, predicts), ARI(labels, predicts)


if __name__ == '__main__':
    Session(__name__).run()
