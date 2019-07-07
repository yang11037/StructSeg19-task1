#!/usr/bin/env python

import argparse
import os
import shutil
import time

from Dataset import CtDataset
#from utils.loss import DiceLoss
from utils.diceloss import DiceLoss
import vnet

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

import setproctitle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=10)
    parser.add_argument('--train-root', type=str)
    parser.add_argument('--dice', action='store_true')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-i', '--inference', default='', type=str, metavar='PATH',
                        help='run inference on data set and save results')

    # 1e-8 works well for lung masks but seems to prevent
    # rapid learning for nodule masks
    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                        metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/vnet.base.{}'.format(datestr())
    weight_decay = args.weight_decay
    if args.train_root == '':
        print("error: please print the data path")
        exit()

    setproctitle.setproctitle(args.save)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    batch_size = args.ngpu*args.batchSz
    model = vnet.VNet(classes=23, batch_size=batch_size)
    gpu_ids = range(args.ngpu)
    model = nn.parallel.DataParallel(model, device_ids=gpu_ids)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if args.cuda:
        model = model.cuda()
        print("cuda done")

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    print("loading training set")
    trainSet = CtDataset(args.train_root, classes=23, mode="train", patch_size=[96, 96, 96],\
                         new_space=[3, 3, 3], winw=350, winl=50)
    trainLoader = data.DataLoader(trainSet, batch_size=batch_size, shuffle=False, **kwargs)
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-1,
                              momentum=0.99, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)
    best_loss = 1
    for epoch in range(1, args.nEpochs + 1):
        print("Epoch {}:".format(epoch))
        train_loss = train(args, epoch, model, trainLoader, optimizer, batch_size)
        is_best = False
        if train_loss < best_loss:
            is_best = True
            best_loss = train_loss
        save_checkpoint({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_prec1': train_loss},
                       is_best, args.save, "vnet_coarse")


def train(args, epoch, model, trainLoader, optimizer, batch_size):
    model.train()
    loss = 0
    for batch_idx, (img, target) in enumerate(trainLoader):
        if args.cuda:
            img, target = img.cuda(), target.cuda()
        target = torch.reshape(target, (batch_size, 23, -1))
        # img, target = Variable(img), Variable(target)
        optimizer.zero_grad()
        output = model(img)
        loss = DiceLoss()(output, target)
        loss.backward()
        optimizer.step()
    print("loss: {}".format(loss))
    return loss


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()



if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()



