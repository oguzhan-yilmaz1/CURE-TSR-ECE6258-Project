import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# from sklearn.svm import SVC
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

parser = argparse.ArgumentParser(description='CURE-TSR Autoencoder training ')

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
parser.add_argument('-f', '--finetune', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--net', default='cnn')

def main():
    global args
    args = parser.parse_args()
    traindir = os.path.join(args.data, 'RealChallengeFree/train')
    testdir = os.path.join(args.data, 'RealChallengeFree/Test')
    train_dataset = utils.CURETSRDataset(traindir, transforms.Compose([
        transforms.Resize([28, 28]), transforms.ToTensor(), utils.l2normalize, utils.standardization]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    test_dataset = utils.CURETSRDataset(testdir, transforms.Compose([
        transforms.Resize([28, 28]), transforms.ToTensor(), utils.l2normalize, utils.standardization]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)


    model = models.AutoEncoder()
    model = torch.nn.DataParallel(model).cuda()
    print("=> creating model %s " % model.__class__.__name__)
    criterion = nn.MSELoss().cuda()

    savedir = 'AutoEncoder'
    checkpointdir = os.path.join('./checkpoints', savedir)
    os.makedirs(checkpointdir, exist_ok=True)
    print('log directory: %s' % os.path.join('./logs', savedir))
    print('checkpoints directory: %s' % checkpointdir)
    logger = Logger(os.path.join('./logs/', savedir))
    if args.evaluate:
        print("=> loading checkpoint ")
        checkpoint = torch.load(os.path.join(checkpointdir, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['AE_state_dict'], strict=False)
        modelCNN = models.Net()
        modelCNN = torch.nn.DataParallel(modelCNN).cuda()
        checkpoint2 = torch.load('./checkpoints/CNN_iter/model_best.pth.tar')
        modelCNN.load_state_dict(checkpoint2['state_dict'], strict=False)
        evaluate(test_loader, model, modelCNN, criterion)
        return
    optimizer = torch.optim.Adam(model.parameters(), args.lr,weight_decay=args.weight_decay)
    cudnn.benchmark = True

    timestart = time.time()

    if args.finetune:
        print("=> loading checkpoint ")
        checkpoint = torch.load(os.path.join(checkpointdir,'model_best.pth.tar'))
        model.load_state_dict(checkpoint['AE_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])


    best_loss = 10e10
    # train_accs = []
    # test_accs = []
    loss_epochs = []

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\n*** Start Training *** \n')
        loss_train= train(train_loader, test_loader, model, criterion, optimizer, epoch)
        print(loss_train)
        loss_epochs.append(loss_train)
        is_best = loss_train < best_loss
        print(best_loss)
        best_loss = min(loss_train, best_loss)
        info = {
            'Loss': loss_train
            # 'Testing Accuracy': test_prec1
        }
        # if not debug:
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch+1)
        if is_best:
            best_epoch = epoch + 1
        save_checkpoint({
            'epoch': epoch + 1,
            'AE_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}, is_best, checkpointdir)
    generate_plots(range(args.start_epoch, args.epochs), loss_epochs)
    print('Best epoch: ', best_epoch)
    print('Total processing time: %.4f' % (time.time() - timestart))
    print('Best loss:', best_loss)
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def evaluate(test_loader, model, modelCNN, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    modelCNN.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        # compute output
        output = modelCNN(model(input_var))
        # loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg
def generate_plots(epochs, train_loss):
    plt.plot(epochs, train_loss, label='loss', color='r')
    # plt.plot(epochs, test_acc, label='test', color='b')
    plt.title("Autoencoder Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./plotss/ChallengeFreeAELoss.png")
def save_checkpoint(state, is_best, checkpointdir):
    fullpath = os.path.join(checkpointdir, 'checkpoint.pth.tar')
    fullpath_best = os.path.join(checkpointdir, 'model_best.pth.tar')
    torch.save(state, fullpath)

    if is_best:
        shutil.copyfile(fullpath, fullpath_best)
def train(train_loader, test_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, ((input, labelin),(target,labelout)) in enumerate(zip(train_loader,test_loader)):
        assert(torch.equal(labelin,labelout))
        data_time.update(time.time() - end)
        # print(input.shape)
        # print(target.shape)
        target = target.cuda(non_blocking=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        # print(output.shape)
        loss = criterion(output, target_var)
        # print(loss.data)
        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data, input.size(0))  # input.size(0): Batch size
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    return losses.avg

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
if __name__ == '__main__':
    main()
