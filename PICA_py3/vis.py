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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

def pickley(filename):
    with open(filename, 'rb') as pickle_file:
        return pickle.load(pickle_file, encoding='latin1')

def pickled(o, path,protocol = -1):
    with open(path, 'wb') as f:
        pickle.dump(o, f, protocol=protocol)

def tsne(myData, y):
    scaler = StandardScaler()
    scaler.fit(myData)
    scaled_data = scaler.transform(myData)
    myData = scaled_data

    tsne = TSNE()
    X_embedded = tsne.fit_transform(myData)
    sns_plot = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full')
    plt.savefig('tsne.png')

def pca(myData, y):
    scaler = StandardScaler()
    scaler.fit(myData)
    scaled_data = scaler.transform(myData)
    myData = scaled_data

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(myData)
    sns_plot = sns.scatterplot(principalComponents[:,0], principalComponents[:,1], hue=y, legend='full')
    plt.savefig('pca.png')

def getTopTwoCats(d):
    m = (-1,0);
    n = (-1,0);
    for cluster in d:
        if(d[cluster] >= m[1]):
            n = m
            m = (cluster, d[cluster])
        elif(d[cluster] >= n[1]):
            n = (cluster, d[cluster])
    return m,n

def kmeanscluster(num_classes,myData,y):
    # kmeans clustering
    kmeans = KMeans(num_classes)
    clusters = kmeans.fit_predict(myData)
    clusterCounts = {}
    for i in range(num_classes):
        clusterCounts[i] = {}
    for i, c in enumerate(clusters):
        #print(f'A: {y[i]}, C: {c}')
        clusterCounts[y[i]][c] = clusterCounts[y[i]].get(c,0) + 1
    for c in clusterCounts:
        ((top,topC),(sec,secC)) = getTopTwoCats(clusterCounts[c])
        print(f'{c}: {top}: {topC/2500.0}, {sec}:{secC/2500.0}')

def tempcluster(myData,y,temps,tempLabels):
    clusters = {}
    #temps = torch.stack(temps).cpu().detach().numpy()
    for i in range(2):
        clusters[i] = []
    for i, data in enumerate(myData):
        pred = None
        minDist = 0
        for j, temp in enumerate(temps):
            currDist = np.linalg.norm(temp - data)
            if(pred == None or currDist < minDist):
                pred = tempLabels[j]
                minDist = currDist
        clusters[pred.item()].append(y[i])
    for i in clusters:
        print(f'{i}: {clusters[i]}')

def main():

    logger.info('Start to declare training variable')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Session will be ran in device: [%s]' % cfg.device)
    start_epoch = 0
    best_acc = 0.

    logger.info('Start to prepare data')

    # testset
    logger.info('testset-------------')
    testset = NewDataSet_test("/home/myz/binary")
    logger.info(len(testset))
    # declare data loaders for testset only
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, 
                                num_workers=cfg.num_workers)
    # testset
    logger.info('tempset-------------')
    tempset = NewDataSet_test("/home/myz/bintemp")
    logger.info(len(tempset))
    # declare data loaders for testset only
    temp_loader = DataLoader(tempset, batch_size=1, shuffle=False, 
                                num_workers=cfg.num_workers)
    
    logger.info('Start to build model')
    net = networks.get()
    torchsummary.summary(net.cuda(), input_size=(1, 32, 32, 32))
    print(net)

    # register hook
    net.heads[0][0].register_forward_hook(get_activation('fc'))

    #criterion = PUILoss(cfg.pica_lamda)
    #optimizer = optimizers.get(params=[val for _, val in net.trainable_parameters().items()])
    #lr_handler = lr_policy.get()

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
    net = net.to(cfg.device)
    #net, criterion = net.to(cfg.device), criterion.to(cfg.device)

    # tensorboard wrtier
    writer = SummaryWriter(cfg.debug, log_dir=cfg.tfb_dir)
    # start training
    #lr = cfg.base_lr
    epoch = start_epoch

    logger.info('Start to evaluate after %d epoch of training' % epoch)
    acc, nmi, ari = evaluate(net, test_loader, temp_loader, testpath="fts.pickle") #'ftstemps.pickle'
    logger.info('Evaluation results at epoch %d are: '
        'ACC: %.3f, NMI: %.3f, ARI: %.3f' % (epoch, acc, nmi, ari))
    writer.add_scalar('Evaluate/ACC', acc, epoch)
    writer.add_scalar('Evaluate/NMI', nmi, epoch)
    writer.add_scalar('Evaluate/ARI', ari, epoch)

def evaluate(net, testloader, temploader, testpath = None, temppath = None):
    """evaluates on provided data
    """
    if(testpath == None):
        net.eval()
        predicts = np.zeros(len(testloader.dataset), dtype=np.int32)
        print('length of predicts: ', len(predicts))
        logger.info('len(testloader.dataset)')
        logger.info(len(testloader.dataset))
        labels = np.zeros(len(testloader.dataset), dtype=np.int32)
      
        myData = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                logger.progress('processing %d/%d batch' % (batch_idx, len(testloader)))
                inputs = inputs.to(cfg.device, non_blocking=True)
                # assuming the last head is the main one
                # output dimension of the last head 
                # should be consistent with the ground-truth
                
                logits = net(inputs)[-1]
                ft = activation['fc']
                myData.append(ft.cpu().detach())
                start = batch_idx * testloader.batch_size
                end = start + testloader.batch_size
                end = min(end, len(testloader.dataset))
                labels[start:end] = targets.cpu().numpy()
                predicts[start:end] = logits.max(1)[1].cpu().numpy()
        myData = torch.cat(myData).numpy()
        
        p = {'x': myData, 'y': labels}
        pickled(p, 'fts.pickle')
    
    else: 
        p = pickley(testpath)
        myData = p['x']
        labels = p['y']
    tsne(myData, labels) 
    pca(myData, labels)
    kmeanscluster(2,myData,labels)

    if(temppath == None):
        net.eval()
        predicts = np.zeros(len(temploader.dataset), dtype=np.int32)
        print('length of predicts: ', len(predicts))
        logger.info('len(temploader.dataset)')
        logger.info(len(temploader.dataset))
        templabels = np.zeros(len(temploader.dataset), dtype=np.int32)

        myTemps = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(temploader):
                logger.progress('processing %d/%d batch' % (batch_idx, len(temploader)))
                inputs = inputs.to(cfg.device, non_blocking=True)
                # assuming the last head is the main one
                # output dimension of the last head 
                # should be consistent with the ground-truth

                logits = net(inputs)[-1]
                ft = activation['fc']
                myTemps.append(ft.cpu().detach())
                start = batch_idx * temploader.batch_size
                end = start + temploader.batch_size
                end = min(end, len(temploader.dataset))
                templabels[start:end] = targets.cpu().numpy()
                predicts[start:end] = logits.max(1)[1].cpu().numpy()
        myTemps = torch.cat(myTemps).numpy()

        p = {'x': myTemps, 'y': labels}
        pickled(p, 'ftstemps.pickle')

    else:
        #myTemps = pickley(temppath)
        #templabels = [0 if x < 2500 else 1 for x in range(5000)]
        p = pickley(temppath)
        myTemps = p['x']
        templabels = p['y']

    tempcluster(myData,labels,myTemps,templabels) 


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
