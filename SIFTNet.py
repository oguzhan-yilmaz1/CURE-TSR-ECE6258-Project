#!/usr/bin/python3


import numpy as np
import pdb
import time
import torch

from torch import nn
from math import pi


class ImageGradientsLayer(torch.nn.Module):


    def __init__(self):
        super(ImageGradientsLayer, self).__init__()

        # Create convolutional layer
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3,
                                bias=False, padding=(1, 1), padding_mode='zeros')

        # Sobel filter
        self.conv2d.weight = get_sobel_xy_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        return self.conv2d(x)


def get_gaussian_kernel(ksize=7, sigma=5) -> torch.nn.Parameter:


    x_cord = torch.arange(ksize)
    x_grid = x_cord.repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = ksize // 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return nn.Parameter(gaussian_kernel)


def get_sobel_xy_parameters() -> torch.nn.Parameter:


    gx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    gy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
    gx.reshape((1, 1, 3, 3))
    gy.reshape((1, 1, 3, 3))


    return torch.nn.Parameter(torch.cat((gx, gy), 0).reshape(2, 1, 3, 3))


"""

SIFT algorithm (See Szeliski 4.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

"""


class HistogramLayer(nn.Module):
    def __init__(self) -> None:

        super(HistogramLayer, self).__init__()

    def forward(self, x) -> torch.Tensor:

        cosines = x[:,:8,:,:] # Contains
        im_grads = x[:,8:,:,:] # Contains dx, dy

        norms = torch.norm(im_grads,dim=1)
        per_px_histogram = torch.zeros_like(cosines)
        for i in range(cosines.shape[2]):
            for j in range(cosines.shape[3]):
                fibre = cosines[:,:,i,j]
                ind = torch.argmax(fibre,1)
                per_px_histogram[:,ind,i,j] += norms[:,i,j]

        return per_px_histogram


class SubGridAccumulationLayer(nn.Module):

    def __init__(self) -> None:

        super(SubGridAccumulationLayer, self).__init__()
        m = nn.Conv2d(8,8,bias=False,padding=(2,2),groups=8, kernel_size=(4,4))
        m.weight = torch.nn.Parameter(torch.ones((8,1,4,4)).float())
        self.layer = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def angles_to_vectors_2d_pytorch(angles: torch.Tensor) -> torch.Tensor:


    angle_vectors=torch.cat((torch.cos(angles).reshape((angles.shape[0],1)),torch.sin(angles).reshape((angles.shape[0],1))),1)

    return angle_vectors


class SIFTOrientationLayer(nn.Module):

    def __init__(self):

        super(SIFTOrientationLayer, self).__init__()

        m = nn.Conv2d(2,10,kernel_size=1,bias=False)
        m.weight = torch.nn.Parameter(self.get_orientation_bin_weights())
        self.layer= m


    def get_orientation_bin_weights(self) -> torch.nn.Parameter():

        angles = torch.tensor([np.pi/8.0,3.0*np.pi/8.0,5.0*np.pi/8.0,7.0*np.pi/8.0,9.0*np.pi/8.0,11.0*np.pi/8.0,13.0*np.pi/8.0,15.0*np.pi/8.0])
        vectors = angles_to_vectors_2d_pytorch(angles)
        vec = torch.cat((vectors,torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]])),0)
        weight_param = vec.reshape(10,2,1,1)

        return weight_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        return self.layer(x)


class SIFTNet(nn.Module):

    def __init__(self):

        super(SIFTNet, self).__init__()

        self.inds = torch.arange(8, 21, 2)
        self.net = nn.Sequential(ImageGradientsLayer(),SIFTOrientationLayer(),HistogramLayer(),SubGridAccumulationLayer())


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # print("sift ",x.shape[0])
        x=self.net(x)
        # print(x.shape, ' 446')
        histogram_grids_per_px = x.detach().permute(0, 2, 3, 1)
        # print(histogram_grids_per_px.shape, ' 448')
        num_interest_pts = self.inds.shape[0]
        fvs = torch.zeros((x.shape[0], num_interest_pts, 128))
        # print(fvs.shape, '451')
        for i, (x_center, y_center) in enumerate(zip(self.inds, self.inds)):
            x = np.linspace(x_center - 6, x_center + 6, 4)
            y = np.linspace(y_center - 6, y_center + 6, 4)
            x_grid, y_grid = np.meshgrid(x, y)

            x_grid = x_grid.flatten().astype(np.int64)
            y_grid = y_grid.flatten().astype(np.int64)
            fvs[:, i, :] = histogram_grids_per_px[:, y_grid, x_grid, :].flatten(start_dim=1)
        # normalize feature vectors to unit length
        fvs /= torch.norm(fvs, dim=2, keepdim=True)
        return torch.pow(fvs, 0.9)
