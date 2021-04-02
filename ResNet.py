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
        self.layerBlock1 = ResidualBlocks(16, n)
        self.layerBlock2 = ResidualBlocks(16, n)
        self.layerBlock3 = ResidualBlocks(16, n)
        self.avgpool = nn.AvgPool2d(kernel_size=16)
        self.fc = nn.Linear(64, 10)
        
    def forward(self, x):
        # Without subsampling
        print('Check0')
        initial_layer = self.initial_layer(x)
        print('Check1')
        resblocks_1 = self.layerBlock1(initial_layer)
        print('Check2')
        resblocks_2 = self.layerBlock2(resblocks_1)
        print('Check3')
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
    
    def __init__(self, channels):
        super().__init__()
        
        input_channels = channels
        output_channels = channels
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = self.conv2(F.relu(self.conv1(x)))
        out = residual + x
        out = F.relu(out)
        return out
    

class ResidualBlocks(nn.Module):
    """
    Sequence of n Residual Blocks of same structure.
    
    """
    def __init__(self, channel, n):
        super().__init__()
        self.n = n
        blocks = []
        for _ in range(n):
            blocks.append(ResidualBlock(channel))
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        out = self.blocks(x)
        return out
    
    

    
def flatten(x):
    N = x.shape[0] 
    return x.view(N, -1) 

