import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimx



class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.
        
        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        self.conv1=nn.Conv2d(num_channels,num_channels,3,1,1,bias=False)
        self.bat1=nn.BatchNorm2d(num_channels)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(num_channels,num_channels,3,1,1,bias=False)
        self.bat2=nn.BatchNorm2d(num_channels)

    def forward(self, x):
        answer=self.conv1(x)
        answer=self.bat1(answer)
        answer=self.relu(answer)
        answer=self.conv2(answer)
        answer=self.bat2(answer)
        answer=answer+x
        answer=self.relu(answer)
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.
        The output should have the same shape as input.
        """
        return answer



class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.conv1=nn.Conv2d(1,num_channels,3,2,1,bias=False)
        self.bat1=nn.BatchNorm2d(num_channels)
        self.relu=nn.ReLU()
        self.max_pool=nn.MaxPool2d(2,2)
        self.block=Block(num_channels)
        self.adpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(num_channels,10)


    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.
        The output should have shape (N, 10).
        """
        answer=self.conv1(x)
        answer=self.bat1(answer)
        answer=self.relu(answer)
        answer=self.max_pool(answer)
        answer=self.block(answer)
        answer=self.adpool(answer)
        answer=answer.view(answer.size(0),-1)
        answer=self.fc(answer)
        return answer