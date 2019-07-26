import torch.nn as nn
import torch as th

from .utils import View
from .blocks import Conv, Linear

class LeNet(nn.Module):
    def __init__(self, bn=True, classes=10, dropout=0., bias=True, **kwargs):
        """Implementation of LeNet [1].

        [1] LeCun Y, Bottou L, Bengio Y, Haffner P. Gradient-based learning applied to
               document recognition. Proceedings of the IEEE. 1998 Nov;86(11):2278-324."""
        super().__init__()

        def conv(ci,co,ksz,psz,dropout=0,bn=True):
            conv_ = Conv(ci,co,kernel_size=ksz, padding=0, bn=bn, bias=bias)

            m = nn.Sequential(
                conv_,
                nn.MaxPool2d(psz,stride=psz),
                nn.Dropout(dropout))
            return m

        self.m = nn.Sequential(
            conv(1,20,(5,5),3,dropout=dropout, bn=bn),
            conv(20,50,(5,5),2,dropout=dropout, bn=bn),
            View(200),
            Linear(200, 500, bn=bn, bias=bias),
            nn.Dropout(dropout),
            Linear(500,classes, bn=bn, bias=bias))

        self.bn = bn

    @property
    def num_parameters(self):
        return sum([w.numel() for w in self.parameters()])

    def forward(self, x):
        x = self.m(x)
        return x
