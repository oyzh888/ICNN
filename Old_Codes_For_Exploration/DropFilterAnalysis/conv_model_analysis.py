import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
import os

class Net(nn.Module):
    def __init__(self,index):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.avgpool = nn.AvgPool2d(32)
        # self.fc = nn.Linear(32 * 32 * 32, 10)
        self.fc = nn.Linear(32, 10)
        self.index = index

    def forward(self, x,labels,epoch):
        res = F.relu(self.conv1(x))
        res = F.relu(self.conv2(res))
        res = F.relu(self.conv3(res))

        # 强制mask掉class1 的filters
        if self.index >= 0:
            label_filter_range = int(res.shape[1] / 10)
            labels = labels * label_filter_range
            label_ones = torch.zeros(res.shape,device=res.device) + 1
            label_ones[:,self.index,:,:] = 0
            label_ones = label_ones.to(x.device)
            res = res * label_ones

        res = self.avgpool(res).squeeze(3).squeeze(2)
        res = self.fc(res)
        return res