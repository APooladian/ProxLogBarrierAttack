import torch as th
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from ProximalLogBarrier.simplex import L1BallProj


class L0NormProx_Batch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,u,x,lbdz):
        proxL0Temp = th.zeros_like(u,device=u.device)
        az = np.sqrt(2)*lbdz.view(-1,1)
        sz = u - x
        boolKeep = abs(sz) > az
        proxL0Temp[boolKeep] = sz[boolKeep]
        return x + proxL0Temp

class LinfNormProx(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,u,x,lbd):
        projL1 = L1BallProj(z=1)
        projTerm, _ = projL1((u - x)/lbd)
        proxInf = u - lbd*projTerm
        return proxInf

class L2NormProx_Batch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,u,x,lbdz):
        lbdz = lbdz.view(-1,1)
        maxTerm = lbdz.clone()
        s = u - x
        normS = s.norm(p=2,dim=-1)
        maxTerm = th.max(lbdz,normS.view(-1,1))
        frac = lbdz / maxTerm
        ProxL2Temp = (1 - frac) * s
        return x + ProxL2Temp

class L1NormProx(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,u,x,lbd):
        s = u-x
        signS = th.sign(s)
        boolKeep = abs(s) > lbd
        SoftT = th.zeros_like(u).cuda()
        SoftT[boolKeep] = signS[boolKeep]
        proxL1 = x + SoftT
        return proxL1
