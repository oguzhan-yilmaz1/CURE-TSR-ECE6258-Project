import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from sklearn.svm import SVC
import argparse
import os
import shutil
import numpy as np

import time
import utils
from utils import AverageMeter
import models
from logger import Logger
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='CURE-TSR Training and Evaluation')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--test', default='', type=str)
parser.add_argument('--loss', default='softmax')
parser.add_argument('--net', default='cnn')

def main():
    global args
    args = parser.parse_args()
    testdir = os.path.join(args.data, 'RealChallengeFree/Test')
    test_dataset = utils.CURETSRDataset(testdir, transforms.Compose([
        transforms.Resize([28, 28]), transforms.ToTensor(), utils.l2normalize, utils.standardization]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    encoder = models.AutoEncoder()


if __name__ == '__main__':
    main()