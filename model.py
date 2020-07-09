import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import utils as vutils
import numpy as np
from PIL import Image
import numpy.random as random
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
from scipy.ndimage import maximum_filter
import pdb


class DecomNet(nn.Module):
    def __init__(self, nfeats=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.conv_0 = nn.Conv2d(4, nfeats//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv   = nn.Conv2d(4, nfeats, kernel_size=3*3, stride=1, padding=4, bias=False)
        self.conv1  = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2  = nn.Conv2d(nfeats, nfeats*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3  = nn.Conv2d(nfeats*2, nfeats*2, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv4  = nn.ConvTranspose2d(nfeats*2, nfeats, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.conv5  = nn.Conv2d(nfeats*2, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv7  = nn.Conv2d(nfeats + nfeats//2, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8  = nn.Conv2d(nfeats, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu =  nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_im):
        input_max, _ = torch.max(input_im, 1, keepdims=True)
        input_im  = torch.cat((input_max, input_im), 1)
        conv_0  = self.relu(self.conv_0(input_im))
        conv = self.conv(input_im)
        conv_1 = self.relu(self.conv1(conv))
        conv_2 = self.relu(self.conv2(conv_1))
        conv_3 = self.relu(self.conv3(conv_2))
        conv_4 = self.relu(self.conv4(conv_3))
        
        conv4_ba2 =  torch.cat((conv_4,conv_1), 1)
        conv_5 = self.relu(self.conv5(conv4_ba2))
        conv_6 =  torch.cat((conv_5,conv_0), 1)
        conv_7 = self.conv7(conv_6)
        conv_8 = self.conv8(conv_7)
        
        R = self.sigmoid(conv_8[:,0:3,:,:])
        L = self.sigmoid(conv_8[:,3:4,:,:])
        
        return R, L

if __name__ == '__main__':
    torch.cuda.set_device(3)
    model = DecomNet().cuda()
    x = Variable(torch.randn(16,3,48,48))
    y = Variable(torch.randn(16,3,48,48))
    out, _ = model(x)
    criterion = nn.L1Loss()
    criterion = criterion.cuda()
    loss = criterion(out, y)
    loss.backward()
