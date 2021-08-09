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
from sklearn.manifold import TSNE

import seaborn as sns
from matplotlib import pyplot as plt
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_palette(["#9b59b6","#3498db"])
palette = sns.color_palette()#("bright", 5)

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
from lib.datasets.new_dataset import NewDataSet, NewDataSet_test
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

activation = {}
def get_activation(name):
    def hook(model, inp, out):
        activation[name] = out#out.detach()
        #print(out[0].shape) # check the size of our feature, which should just be 1024
    return hook

def tsne(myData, y):
    tsne = TSNE()
    X_embedded = tsne.fit_transform(myData)
    sns_plot = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full')
    plt.savefig('output.png')

def main():

    logger.info('Start to declare training variable')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Session will be ran in device: [%s]' % cfg.device)
    start_epoch = 0
    best_acc = 0.

    logger.info('Start to prepare data')

    # otrainset: original trainset
    logger.info('otrainset-------------')
    otrainset = [NewDataSet()]
    logger.info(len(otrainset[0]))

    # ptrainset: perturbed trainset
    logger.info('ptrainset-------------')
    ptrainset = [NewDataSet()]
    logger.info(len(ptrainset[0]))

    # testset
    logger.info('testset-------------')
    testset = NewDataSet_test()
    logger.info(len(testset))
    # declare data loaders for testset only
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, 
                                num_workers=cfg.num_workers)
    logger.info('Start to build model')
    net = networks.get()
    torchsummary.summary(net.cuda(), input_size=(1, 32, 32, 32))
    print(net)
    

    # register hook
    net.heads[0][0].register_forward_hook(get_activation('fc'))


    criterion = PUILoss(cfg.pica_lamda)
    optimizer = optimizers.get(params=[val for _, val in net.trainable_parameters().items()])
    lr_handler = lr_policy.get()

    # load session if checkpoint is provided
    #cfg.resume = 'sessions/20210802-231758/checkpoint/latest.ckpt'
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

    logger.info('Start to evaluate after %d epoch of training' % epoch)
    acc, nmi, ari = evaluate(net, test_loader)
    logger.info('Evaluation results at epoch %d are: '
        'ACC: %.3f, NMI: %.3f, ARI: %.3f' % (epoch, acc, nmi, ari))
    writer.add_scalar('Evaluate/ACC', acc, epoch)
    writer.add_scalar('Evaluate/NMI', nmi, epoch)
    writer.add_scalar('Evaluate/ARI', ari, epoch)

def evaluate(net, loader):
    """evaluates on provided data
    """

    net.eval()
    predicts = np.zeros(len(loader.dataset), dtype=np.int32)
    print('length of predicts: ', len(predicts))
    logger.info('len(loader.dataset)')
    logger.info(len(loader.dataset))
    labels = np.zeros(len(loader.dataset), dtype=np.int32)
  
    myData = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            logger.progress('processing %d/%d batch' % (batch_idx, len(loader)))
            inputs = inputs.to(cfg.device, non_blocking=True)
            # assuming the last head is the main one
            # output dimension of the last head 
            # should be consistent with the ground-truth
            
            logits = net(inputs)[-1]
            ft = activation['fc']
            myData.append(ft.cpu().detach())
            start = batch_idx * loader.batch_size
            end = start + loader.batch_size
            end = min(end, len(loader.dataset))
            labels[start:end] = targets.cpu().numpy()
            predicts[start:end] = logits.max(1)[1].cpu().numpy()
    myData = torch.cat(myData).numpy()
    tsne(myData, labels) 

    #print(f'predicts: {predicts}, labels: {labels}')
    # compute accuracy
    num_classes = 6 #labels.max().item() + 1
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
