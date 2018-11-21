import torch.nn as nn
from ConvLayer import ConvLayer

class ResUnit(nn.Module):
    def __init__(self, channels, kernel):
        super(ResUnit,self).__init__()
        self.bn1 = nn.BatchNorm2d(channels,affine=True)
        self.conv1 = ConvLayer(channels,channels,kernel,stride=1)
        
        self.bn2 = nn.BatchNorm2d(channels,affine=True)
        self.conv2 = ConvLayer(channels,channels,kernel,stride=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = out + x
        
        return out
