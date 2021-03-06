#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 5)


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.data import TensorDataset, DataLoader

import moco.loader
import moco.builder

import Encoder3D.Model_RB3D
import Encoder3D.Model_DSRF3D_v2
import Custom_CryoET_DataLoader
from CustomTransforms import ToTensor, Random3DRotate

import torchio as tio
os.environ['OPENBLAS_NUM_THREADS'] = '1'
model_names = ['RB3D', 'DSRF3D_v2']

evalOnTest = True
Encoders3D_dictionary = {'RB3D': Encoder3D.Model_RB3D.RB3D, 'DSRF3D_v2':Encoder3D.Model_DSRF3D_v2.DSRF3D_v2}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RB3D',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: RB3D)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=7, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=128, type=int,
                    help='queue size; number of negative keys (default: 100)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print(f'gpuspernode{ngpus_per_node}')        
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    print(f'actually starting process main worker, with gpu={gpu}')
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
           args.rank = args.rank * ngpus_per_node + gpu
    # os.environ['MASTER_ADDR'] = 'localhost'    
    # os.environ['MASTER_PORT'] = '12355'        
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    print(type(args.arch))
    model = moco.builder.MoCo(
        Encoders3D_dictionary[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)
    net = moco.builder.Net()
    print(net)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model.encoder_q.fc2.register_forward_hook(get_activation('fc2')) 
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
                # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'subtomogram_mrc')
    traindir_json = os.path.join(args.data, 'json_label')
    dataTest = "/home/myz/data3_SNRinfinity"    
    testdir = os.path.join(dataTest, 'subtomogram_mrc')
    testdir_json = os.path.join(dataTest, 'json_label')
    dataTemp = "/home/myz/temps_10"    
    tempdir = os.path.join(dataTemp, 'subtomogram_mrc')
    tempdir_json = os.path.join(dataTemp, 'json_label')
    #testdir = traindir
    #testdir_json = traindir_json 
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     # std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            tio.transforms.RandomFlip(flip_probability=0.6),
            tio.transforms.RandomGamma(log_gamma=(-0.3, 0.3), p=0.7),
            tio.transforms.RandomBlur(p=0.5)
            #tio.transforms.RandomGamma()
            # tio.transforms.RandomAffine(),
            # tio.transforms.RandomSwap(num_iterations=5),
            # tio.transforms.ZNormalization()
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            tio.transforms.RandomFlip(flip_probability=0.6),
            tio.transforms.RandomAffine(degrees=45,translation=0.1),
 
            #ToTensor(),
            #transforms.RandomResizedCrop(32, scale=(0.5, 1.), ratio=(1.,1.)),
            #transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.9, 1.1))
            
            #ToTensor()
            # tio.transforms.RandomFlip(flip_probability=0.5, axes=(0,1,2)),
            #tio.transforms.RandomFlip(flip_probability=0.5),
            # tio.transforms.RandomBlur(p=0.5)
            #tio.transforms.RandomNoise()
            #tio.transforms.RandomGamma()
            # tio.transforms.RandomAffine(),
            # tio.transforms.RandomSwap(num_iterations=5),
            # tio.transforms.ZNormalization()
        ]
   
    train_dataset = Custom_CryoET_DataLoader.CryoETDatasetLoader(
        root_dir = traindir, json_dir = traindir_json,
        transform = moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    #if args.distributed:
    #    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #else:
    #    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True,  drop_last=True)


    test_dataset = Custom_CryoET_DataLoader.CryoETDatasetLoader(
        root_dir = testdir, json_dir = testdir_json,
        transform = moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, 
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
       
    temp_dataset = Custom_CryoET_DataLoader.CryoETDatasetLoader(
        root_dir = tempdir, json_dir = tempdir_json,
        transform = moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    temp_loader = torch.utils.data.DataLoader(
        temp_dataset, batch_size=1, 
        num_workers=args.workers, pin_memory=True, drop_last=True)


    cluster(train_loader, test_loader, temp_loader, model, net, criterion, optimizer, 200, args)

activation = {}
def get_activation(name):
    def hook(model, inp, out):
        activation[name] = out#out.detach()
        # print(out[0].shape) # check the size of our feature, which should just be 1024
    return hook

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

def tsne(myData, y):
    tsne = TSNE()
    X_embedded = tsne.fit_transform(myData)
    sns_plot = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full')
    plt.savefig('output.png') 

def linear(net, myData, y):
    y = y - 1
    dataset = TensorDataset(torch.Tensor(myData), torch.LongTensor(y))
    loader = DataLoader(
        dataset,
        batch_size=2
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

 
def cluster(train_loader,test_loader, temp_loader, model, net, criterion, optimizer, epoch, args):
    num_classes = 2
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.eval()
    #model.register_forward_hook(get_activation('fc2')) 
    
    end = time.time()
    myData = []
    y = []
    print('startin')
    for i, (images, label) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        #print(f'images[0]: {images[0].shape}, 1: {images[1].shape}')
        output, target = model(im_q=images[0], im_k=images[1])
        ft = activation['fc2']
        
        myData.append(ft[0])
        y.append(label[0])
        myData.append(ft[1])
        y.append(label[1])
        
        #loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #losses.update(loss.item(), images[0].size(0))
        #top1.update(acc1[0], images[0].size(0))
        #top5.update(acc5[0], images[0].size(0))

        # measure elapsed time
        #batch_time.update(time.time() - end)
        #end = time.time()

        #if i % args.print_freq == 0:
        #    progress.display(i)

    myData = torch.stack(myData).cpu().detach().numpy()
    print(myData.shape)
    y = torch.stack(y).cpu().detach().numpy()


    #linear(net, myData,y)

    # TSNE visualization
    tsne(myData, y)
    
    # kmeans clustering
    kmeans = KMeans(num_classes)
    clusters = kmeans.fit_predict(myData)
    clusterCounts = {}
    for i in range(10):
        clusterCounts[i] = {}
    for i, c in enumerate(clusters):
        print(f'A: {y[i]}, C: {c}')
        clusterCounts[y[i].item()][c.item()] = clusterCounts[y[i].item()].get(c.item(),0) + 1
    for c in clusterCounts:
        ((top,topC),(sec,secC)) = getTopTwoCats(clusterCounts[c])
        print(f'{c}: {top}: {topC/50.0}, {sec}:{secC/50.0}')
        #print(f'{c}: {clusterCounts[c][c]/50.0}, {clusterCounts[c]}')
    
    temps = []
    tempLabels = []
    for i, (images, label) in enumerate(temp_loader):
        # measure data loading time
        #data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        ft = activation['fc2']
        temps.append(ft[0])
        tempLabels.append(label[0])
    clusters = {}
    temps = torch.stack(temps).cpu().detach().numpy()
    for i in range(10):
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

def train(train_loader, model, criterion, optimizer, epoch, args, isTrain):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if(isTrain): model.train()
    else: model.eval()
    
    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
         
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        if (isTrain):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='main_moco_checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'main_moco_checkpoint/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        #print(output)
        _, pred = output.topk(maxk, 1, True, True)
        #print(pred)
        pred = pred.t()
        #print(f'pred: {pred}, actual: {target}')        
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        #print(correct)
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

