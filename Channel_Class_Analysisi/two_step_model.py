
from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math
import torch
from torch.distributions import Bernoulli
import torch.nn.functional as F

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class LearnableMaskLayer(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LearnableMaskLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mask = torch.nn.Parameter(torch.full((64,num_classes),0.5))

    def get_channel_mask(self):
        c_mask = torch.relu(self.mask)
        return c_mask

    def _icnn_mask(self, x, labels, epoch):
        # if (epoch % 3 == 1 and self.training):
        #     index_mask = torch.zeros(x.shape, device=x.device)
        #     for idx, la in enumerate(labels):
        #         index_mask[idx, :, :, :] = self.mask[:, la].view(-1, self.mask.shape[0], 1, 1)
        #     return index_mask * x
        # else:
        #     return x

        # mask some channels
        lmask = self.get_channel_mask()
        mask_zeros = torch.zeros(lmask[:, 0].shape, device=x.device)
        # mask_zeros += self.mask[:, 5] # mask first class channels
        # mask_zeros += self.mask[:, 1]  # mask first class channels
        mask_zeros += torch.sum(lmask[:], dim=-1)
        mask_zeros -= lmask[:, 1]  # mask first class channels
        mask_zeros -= lmask[:, 5]  # mask first class channels

        return mask_zeros.view(-1,mask_zeros.shape[0],1,1) * x

        # return x

    def loss_function(self):
        # L1 regulization
        lMask = self.get_channel_mask()
        lambda1 = 1e-4
        l1_regularization = lambda1 * torch.norm(lMask, 1)
        # import ipdb; ipdb.set_trace()

        # class_max[0]->elements; class_max[1] -> indices
        class_max = torch.max(lMask, dim=1)
        # Only one class is expected for one channel
        structure_loss = torch.norm((1 - class_max[0]), 1) * 1e-3
        return structure_loss + l1_regularization

    def forward(self, x, labels, epoch, last_layer_mask=None):
        if (last_layer_mask is not None):
            self.last_layer_mask = last_layer_mask

        x = self._icnn_mask(x, labels, epoch)

        return x, self.loss_function()


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        # self.mask = torch.nn.Parameter(torch.full((64,num_classes),0.5))
        self.lmask = LearnableMaskLayer(feature_dim=64, num_classes=num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, labels, epoch):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        # x, reg = self.lmask(x, labels, epoch)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # print('l1_regularization, ', l1_regularization, 'l2_regularization', l2_regularization)

        return x

def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)