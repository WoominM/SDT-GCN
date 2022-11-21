import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class PointWise_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(PointWise_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              1, 
                              stride=(stride, 1), 
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class ScaleWise_GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_scale=4):
        super(ScaleWise_GraphConv, self).__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.Nh = A.size(0)
        self.A = nn.Parameter(A)
        
        self.num_scale = num_scale
        rel_channels = in_channels // 8 if in_channels != 3 else 8
        
        self.convV = PointWise_Conv(in_channels, out_channels * self.Nh, 1, groups=num_scale)
        self.convQK = nn.Conv2d(in_channels, 
                                rel_channels * 2 * self.Nh, 
                                1, 
                                groups=num_scale)
        self.convW = nn.Conv2d(rel_channels * self.Nh, 
                               out_channels * self.Nh, 
                               1, 
                               groups=num_scale * self.Nh)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
    
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x
        v = self.relu(self.convV(x)).view(N, self.num_scale, self.Nh, -1, T, V)
        dtype, device = v.dtype, v.device
        
        # weight
        A = self.A.to(dtype)
        q, k = self.convQK(x).view(N, self.num_scale, self.Nh, -1, T, V).chunk(2, 3)
        qm, km = q.mean(-2).view(N, -1, V, 1), k.mean(-2).view(N, -1, 1, V)
        weights = self.convW(self.tanh(qm - km)).view(N, self.num_scale, self.Nh, -1, V, V) #nhcvv
        weights = A.view(1, 1, self.Nh, 1, V, V) + self.alpha.to(dtype) * weights
        
        # aggregation
        x = torch.einsum('nshcvu,nshctu->nsctv', weights, v).view(N, -1, T, V)
        x = self.bn(x)
        return x