import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.avgpool = nn.AvgPool2d(32)
        # self.fc = nn.Linear(32 * 32 * 32, 10)
        self.fc = nn.Linear(32, 10)

    def forward(self, x,labels,epoch):
        res = F.relu(self.conv1(x))
        res = F.relu(self.conv2(res))
        res = F.relu(self.conv3(res))

        res_icnn = res
        label_filter_range = int(res_icnn.shape[1] / 10)
        if (int(epoch) % 3 == 0 and self.training and epoch>20):
            labels = labels * label_filter_range
            label_ones = torch.zeros(res_icnn.shape,device=res.device) + 1e-2
            for idx, la in enumerate(labels):
                label_ones[idx, la:la + label_filter_range, :, :] = 1
            res_icnn = res_icnn * label_ones.to(x.device)

        res = self.avgpool(res).squeeze(3).squeeze(2)
        res_icnn = self.avgpool(res_icnn).squeeze(3).squeeze(2)
        # res = res.view(res.shape[0], 32 * 32 * 32)
        res = self.fc(res)
        res_icnn = self.fc(res_icnn)
        return res,res_icnn