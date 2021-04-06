import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F

import numpy as np

class ResNet(nn.Module):
    
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.initial_layer = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.initial_layer.weight)
        self.layerBlock1 = ResidualBlocks(16, 16, n)
        self.layerBlock2 = ResidualBlocks(16, 32, n)
        self.layerBlock3 = ResidualBlocks(32, 64, n)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(64, 10)
        nn.init.kaiming_normal_(self.fc.weight)
        
    def forward(self, x):
        initial_layer = self.initial_layer(x)
        resblocks_1 = self.layerBlock1(initial_layer)
        resblocks_2 = self.layerBlock2(resblocks_1)
        resblocks_3 = self.layerBlock3(resblocks_2)
        avg_pool = self.avgpool(resblocks_3)
        flat = flatten(avg_pool)
        scores = self.fc(flat)
        return scores
        
        
class ResidualBlock(nn.Module):
    """
    Single Residual Block.
    
    Structure:
    conv - relu - conv - identity - relu
    """
    
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.sub_stride = 1
        
        # layers for subsampling
        if input_channels != output_channels: 
            self.sub_stride = 2
            self.max = nn.MaxPool2d(1, stride=self.sub_stride)
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=self.sub_stride, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        
        nn.init.kaiming_normal_(self.conv1.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, a=0, mode='fan_in', nonlinearity='relu')
        
        
    def forward(self, x):
        residual = self.conv2(F.relu(self.conv1(x)))
        
        # resize x by zero padding and downsampling via maxpoling to match dimensions of feature maps
        if residual.shape[1] != x.shape[1]:
            pad = int((residual.shape[1] - x.shape[1])/2)
            x = F.pad(x, (0, 0, 0, 0, pad, pad, 0, 0))
            x = self.max(x)
            
        out = residual + x
        out = F.relu(out)
        return out
    

class ResidualBlocks(nn.Module):
    """
    Sequence of n Residual Blocks of same structure.
    
    """
    def __init__(self, input_channels, output_channels, n):
        super().__init__()
        #self.n = n
        blocks = [ResidualBlock(input_channels, output_channels)]
        for _ in range(n-1):
            blocks.append(ResidualBlock(output_channels, output_channels))
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        out = self.blocks(x)
        return out
    
    

    
def flatten(x):
    N = x.shape[0] 
    return x.view(N, -1) 

