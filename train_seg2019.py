#!/usr/bin/env python

import argparse

from Dataset import CtDataset
#from utils.loss import DiceLoss
from utils.diceloss import DiceLoss
import vnet

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batchSz', type=int, default=10)
    # parser.add_argument('--dice', action='store_true')
    # parser.add_argument('--ngpu', type=int, default=1)
    # parser.add_argument('--nEpochs', type=int, default=300)
    # parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
    #                     help='manual epoch number (useful on restarts)')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
    #                     help='evaluate model on validation set')
    # parser.add_argument('-i', '--inference', default='', type=str, metavar='PATH',
    #                     help='run inference on data set and save results')
    #
    # # 1e-8 works well for lung masks but seems to prevent
    # # rapid learning for nodule masks
    # parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
    #                     metavar='W', help='weight decay (default: 1e-8)')
    # parser.add_argument('--no-cuda', action='store_true')
    # parser.add_argument('--save')
    # parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--opt', type=str, default='adam',
    #                     choices=('sgd', 'adam', 'rmsprop'))
    # args = parser.parse_args()
    trainSet = CtDataset("./data/test", classes=23, mode="train", patch_size=[96, 96, 96],\
                         new_space=[3, 3, 3], winw=350, winl=50)
    trainLoader = data.DataLoader(trainSet, batch_size=1, shuffle=False)
    model = vnet.VNet(classes=23, batch_size=1).cpu()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-8)
    for epoch in range(1, 100):
        model.train()
        for batch_idx, (img, target) in enumerate(trainLoader):
            target = torch.reshape(target, (1, 23, -1))
            #img, target = Variable(img), Variable(target)
            optimizer.zero_grad()
            output = model(img)
            loss = DiceLoss()(output, target)
            print(loss)
            loss.backward()
            optimizer.step()
            print("Epoch {}, batchidx {}, loss: {}".format(epoch, batch_idx, loss))


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()



