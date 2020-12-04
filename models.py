import torch
import torch.nn as nn
import torch.nn.functional as F
from SIFTNet import SIFTNet
import numpy as np
from utils import rgb2gray
from skimage.feature import hog
# class Hog():
#     def __init__(self):
#         super(Hog, self).__init__()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # self.encconv1 = nn.Conv2d(3, 6, 3)
        # # self.pool = nn.MaxPool2d(2, 2)
        # self.encconv2 = nn.Conv2d(6, 16, 5)
        # # self.conv2d_drop = nn.Dropout()
        # self.enclin1 = nn.Linear(16 * 4 * 4, 120)
        # # self.enclin2 = nn.Linear(120, 14)
        # # self.declin2 = nn.Linear(14, 120)
        # self.declin1 = nn.Linear(120,16 * 4 * 4)
        # self.decconv2 = nn.ConvTranspose2d(16,6,5)
        # self.deconv1 = nn.ConvTranspose2d(6,3,3)
        self.encconv1 = nn.Conv2d(3,16,3,padding=1)
        self.encpool1 = nn.MaxPool2d(2, 2)
        self.encconv2 = nn.Conv2d(16,8,3,padding=1)
        self.encpool2 = nn.MaxPool2d(2, 2)
        self.decconv1 = nn.ConvTranspose2d(8,16,2,stride=2)
        self.decconv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        # x = F.relu(self.encconv1(x))
        # print(x.shape)
        # x = F.relu(self.encconv2(x))
        # print(x.shape)
        # x = F.relu(self.enclin1(x))
        # print(x.shape)
        # x = F.relu(self.declin1(x))
        # x = F.relu(self.decconv2(x))
        # x = F.relu(self.decconv1(x))
        x = F.relu(self.encconv1(x))
        x = self.encpool1(x)
        x = F.relu(self.encconv2(x))
        x = self.encpool2(x)
        x = F.relu(self.decconv1(x))
        x = F.sigmoid(self.decconv2(x))
        return x

class EncodeCNN(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encconv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.encpool1 = nn.MaxPool2d(2, 2)
        self.encconv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.encpool2 = nn.MaxPool2d(2, 2)
        self.classify1 = nn.Linear(120, 84)
        self.classify2 = nn.Linear(84, 14)
        # self.cnndrop = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.encconv1(x))
        print(x.shape)
        x = F.relu(self.encconv2(x))
        print(x.shape)
        x = F.relu(self.enclin1(x))
        x = F.relu(self.classify1(x))
        # x = self.cnndrop(x)
        x = self.classify2(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv2d_drop = nn.Dropout()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 14)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2d_drop(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv2d_drop(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RGBClassifier(nn.Module):
    def __init__(self):
        super(RGBClassifier, self).__init__()
        self.fc = nn.Linear(28 * 28 * 3, 14, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_().mul_(0.005)

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 3)
        x = self.fc(x)
        return x

## BEGIN MY OWN IMPLEMENTATIONS

class IntensityClassifier(nn.Module):
    def __init__(self):
        super(IntensityClassifier, self).__init__()
        self.fcI = nn.Linear(28 * 28, 14, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_().mul_(0.005)

    def forward(self, x):
        x = x.mean(dim=1)
        x = x.view(-1, 28 * 28)
        # print(x.requires_grad)
        x = self.fcI(x)
        return x

class SIFTClassifier(nn.Module):
    def __init__(self):
        super(SIFTClassifier, self).__init__()
        self.sift = SIFTNet().cuda()
        self.fcS = nn.Linear(7*128, 14, bias=False)
        #
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_().mul_(0.005)

    def forward(self, x):
        # print(x.shape,' 74')
        x=rgb2gray(x).unsqueeze(1)
        # print(x.shape, ' 76')
        x=self.sift(x).cuda()
        # print(x.shape, ' 78')
        # xc = x.clone()
        # # print(xc.requires_grad)
        x = x.view(-1, 7*128)
        # print(x.shape, ' 82')
        x = self.fcS(x)
        # print(x.shape, ' 84')
        return x
