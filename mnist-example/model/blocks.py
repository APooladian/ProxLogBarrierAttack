import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import math

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim,  bn=True, dropout=0.,
            bias=True, **kwargs):
        """A linear block.  The linear layer is followed by batch
        normalization (if active) and a ReLU (again, if active)

        Args:
            in_dim: number of input dimensions
            out_dim: number of output dimensions
            bn (bool, optional): turn on batch norm (default: False)
        """
        super().__init__()

        self.weight = nn.Parameter(th.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(th.randn(out_dim)) 
        else:
            self.register_parameter('bias', None)
        self.out_dim = out_dim
        self.in_dim = in_dim
        if bn:
            self.bn = nn.BatchNorm1d(out_dim, affine=False)
        else:
            self.bn = False

        if dropout>0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        self.reset_parameters()


    def reset_parameters(self):
        n = self.in_dim
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)

        y = F.linear(x, self.weight, None)

        if self.bn:
            y = self.bn(y)

        if self.bias is not None:
            b = self.bias.view(1,-1)
            y = y+b

        y = F.relu(y)

        return y

    def extra_repr(self):
        s = ('{in_dim}, {out_dim}')

        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'

        return s.format(**self.__dict__)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels,  stride=1, padding=None,
            kernel_size=(3,3),  bn=True, dropout = 0.,bias=True,**kwargs):
        """A 2d convolution block.  The convolution is followed by batch
        normalization (if active).

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride (int, optional): stride of the convolutions (default: 1)
            kernel_size (tuple, optional): kernel shape (default: 3)
            bn (bool, optional): turn on batch norm (default: False)
        """
        super().__init__()

        self.weight = nn.Parameter(th.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(th.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        self.stride = stride
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size=_pair(kernel_size)
        if padding is None:
            self.padding = tuple([k//2 for k in kernel_size])
        else:
            self.padding = _pair(padding)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels, affine=False)
        else:
            self.bn = False

        if dropout>0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)

        y = F.conv2d(x, self.weight, None, self.stride, self.padding,
                1, 1)

        if self.bn:
            y = self.bn(y)

        if self.bias is not None:
            b = self.bias.view(1,self.out_channels,1,1)
            y = y+b

        y = F.relu(y)

        return y

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'
        return s.format(**self.__dict__)
