import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
import os

class LearnableMaskLayer(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LearnableMaskLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mask = torch.nn.Parameter(torch.full((feature_dim,num_classes),0.5))

    def get_channel_mask(self):
        c_mask = self.mask
        return c_mask

    def _icnn_mask(self, x, labels, epoch):
        if (epoch % 3 == 1 and self.training):
            index_mask = torch.zeros(x.shape, device=x.device)
            for idx, la in enumerate(labels):
                index_mask[idx, :, :, :] = self.mask[:, la].view(-1, self.mask.shape[0], 1, 1)
            return index_mask * x
        else:
            return x

    def loss_function(self):
        # L1 regulization
        # lMask dim: (64,num_classes)
        lMask = self.get_channel_mask()

        # L1 reg
        lambda1 = 1e-3
        l1_reg= lambda1 * torch.norm(lMask, 1)
        # import ipdb; ipdb.set_trace()

        # L_max reg
        # after torch.max(), class_max[0]->elements; class_max[1] -> indices
        class_max = torch.max(lMask, dim=1)
        # Only one class is expected for one channel
        l_max_reg = torch.norm(class_max[0], 1) * 1e-3

        # L_21 reg
        l_21_reg = torch.norm(torch.norm(lMask, 1, dim=1), 2) * 1e-3

        # Related the max of 30% of the classes:

        # import ipdb; ipdb.set_trace()
        # idx = torch.topk(-lMask, k=int(lMask.shape[1]*0.7), dim=1)[1]
        # test_arr = torch.arange(0, 64*10, device=lMask.device).resize_((64, 10))
        # idx = idx.view(idx.shape[0],idx.shape[1],1)
        # test_arr[idx.view(-1,1)]
        # test_arr.scatter_(dim=0, index=idx, src=0)
        # import ipdb;
        # ipdb.set_trace()
        activate_max_num_reg = torch.norm(lMask, 1, dim=1) - (lMask.shape[1]*0.3)
        activate_max_num_reg = torch.relu(activate_max_num_reg)
        activate_max_num_reg = torch.sum(activate_max_num_reg)* lambda1


        # print(l_max_reg , l1_reg , l_21_reg)
        return l1_reg + activate_max_num_reg
        return l_max_reg + l1_reg + l_21_reg

    def clip_lmask(self):

        lmask = self.mask
        lmask = lmask / torch.max(lmask, dim=1)[0].view(-1, 1)
        lmask = torch.clamp(lmask, min=0, max=1)
        self.mask.data = lmask

    def forward(self, x, labels, epoch, last_layer_mask=None):
        if (last_layer_mask is not None):
            self.last_layer_mask = last_layer_mask

        x = self._icnn_mask(x, labels, epoch)

        return x, self.loss_function()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.avgpool = nn.AvgPool2d(32)
        # self.fc = nn.Linear(32 * 32 * 32, 10)
        self.fc = nn.Linear(32, 10)
        self.lmask = LearnableMaskLayer(feature_dim=32, num_classes=10)


    def forward(self, x,labels,epoch):
        res = F.relu(self.conv1(x))
        res = F.relu(self.conv2(res))
        res = F.relu(self.conv3(res))

        res, reg = self.lmask(res, labels, epoch)

        res = self.avgpool(res).squeeze(3).squeeze(2)
        res = self.fc(res)
        return res,reg