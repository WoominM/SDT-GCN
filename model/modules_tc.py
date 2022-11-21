import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1, pad=True):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2 if pad else 0
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x.shape
        x = self.unfold(x)
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(N, C, self.window_size, -1, V).permute(0, 1, 3, 2, 4).contiguous()
        return x

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class TemporalAttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_scale=1, stride=1, dilation=1):
        super(TemporalAttentionConv, self).__init__()

        rel_reduction = 8
        rel_channels = in_channels // rel_reduction if in_channels != 3 else 8
        
        self.num_scale = num_scale
        self.ks = kernel_size
        self.unfold = UnfoldTemporalWindows(window_size=self.ks, 
                                            window_stride=stride, 
                                            window_dilation=dilation)
        self.convV = nn.Conv2d(in_channels, rel_channels, 1, groups=self.num_scale) 
        self.convW = nn.Conv2d(self.ks, self.ks**2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(rel_channels, out_channels, 1, stride=1),
            nn.BatchNorm2d(out_channels))
        
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, rel_channels, 1, stride=(stride, 1), groups=self.num_scale),
            nn.BatchNorm2d(rel_channels),
        ) if in_channels != out_channels or stride != 1 else (lambda x: x)
        self.bn = nn.BatchNorm2d(rel_channels)
        
        self.relu = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        res = x
        x, v = self.unfold(x), self.unfold(self.convV(x)) #nctwv
        N, C, T, W, V = x.size()
        
        #weight
        x = x.mean(1).transpose(1, 2).contiguous()
        weights = self.tanh(self.convW(x).view(N, W, W, T, V))
        
        #aggregation
        x = torch.einsum('nwutv,nctuv->nctv', weights, v)
        x = self.relu(self.bn(x) + self.residual(res))
        x = self.conv(x)
        return x   

class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4]):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.LeakyReLU(0.1),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation),
            )
            for dilation in dilations
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))
        

    def forward(self, x):
        # Input dim: (N,C,T,V)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        return out