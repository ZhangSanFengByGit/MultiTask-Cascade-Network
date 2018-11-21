

import torch.nn as nn
from ResUnit import ResUnit

class mask_generate(nn.Module):
	def __init__(self):
		super(mask_generate , self).__init__()
		self.res1 = ResUnit(channels = 512, kernel = 3)   #size 61*61
		self.res2 = ResUnit(channels = 512, kernel = 3)
		self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2)   #size 30*30
		self.conv = nn.Conv2d(512,512,kernel_size = 4, stride = 2)   #size 14*14
		self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)   #size 7*7
		self.fc1 = nn.Linear(7*7*512, 128*64)
		self.fc2 = nn.Linear(128*64, 64*64)
		self.sigmoid = nn.Sigmoid()

	def forward(self,x):
		x = self.res2(self.res1(x))
		x = self.pool(x)
		x = self.pool2(self.conv(x))
		x = x.view(1,-1)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		x = x.view(64,64)
		
		return x

