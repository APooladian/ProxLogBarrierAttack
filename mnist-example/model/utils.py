import torch.nn as nn
import torch as th
from torch.nn import functional as F

# Return the flattened array
class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)
