import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli
from torch.autograd import Variable


class SENet(nn.Module):

    def __init__(self,inplanes=64,h=32,):
        super(SENet, self).__init__()
        self.h = h
        self.inplanes = inplanes
        #bottle net
        self.W1 = torch.nn.Parameter(torch.randn(h, inplanes))
        self.W2 = torch.nn.Parameter(torch.randn(inplanes, h))

    def normalize(self, input):
        N,C,H,W = input.shape
        input_vec = input.view(N,C,H*W)
        min_c, max_c = input_vec.min(2, keepdim=True)[0], input_vec.max(2, keepdim=True)[0]
        min_c, max_c = min_c.view(N,C,1,1),max_c.view(N,C,1,1)
        input_norm = (input - min_c) / (max_c - min_c + 1e-8)
        return input_norm

    def gather(self,input):
        # N,C,H,W = input.shape
        # input_vec = input.view(N,C,H*W)
        gather_vec = input.mean(dim=-1)
        return gather_vec

    def excite(self,x):
        #N,C,1
        x1 = F.relu(torch.matmul(self.W1.view(1,self.h,self.inplanes),x))
        out = torch.matmul(self.W2.view(1,self.inplanes,self.h),x1)
        out = F.sigmoid(out)
        return out

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"
        N, C, H, W = x.shape
        g = x.view(N,C,H*W)
        g = self.gather(g).view(N, C, 1)  # N*C*1
        spatial_mask = self.excite(g)
        # print (spatial_mask)
        out = x * spatial_mask.view(N, C, 1, 1)
        return out,spatial_mask

