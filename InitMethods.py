import argparse
import os, sys

import numpy as np
import pandas as pd
import pickle as pk
import random

import torch
from torch import nn
import torch.nn.functional as F

import torchvision.models as models

class UniformInitialize():

    def __init__(self, model):
        self.model = model

    def __call__(self, x, y, criterion, max_iters=1e4, bounds=(0,1)):
        xpert = x.clone()
        dt=0.01
        Unif = torch.distributions.Uniform(-1,1)
        nbd_evals = 0
        correct = criterion(xpert,y)
        nbd_evals += 1
        k=0
        while correct.sum() > 0:

            l = correct.sum()
            USample = Unif.sample((l,*x.shape[1:]))
            if torch.cuda.is_available():
                USample = USample.cuda()
            xpert[correct] = x[correct] + 1.01**k*dt*USample
            xpert.clamp_(*bounds)
            correct = criterion(xpert,y)
            nbd_evals += 1
            k+=1

        return xpert


class GaussianInitialize():

    def __init__(self, model):
        self.model = model

    def __call__(self, x, y, criterion, max_iters=1e4, bounds=(0,1)):
        xpert = x.clone()
        dt = 0.01
        correct = criterion(xpert,y)
        k=0
        while correct.sum()>0:
            l = correct.sum()
            xpert[correct] = x[correct] + (1.01)**k*dt*torch.randn(l,*xpert.shape[1:],
                                                        device=xpert.device)
            xpert.clamp_(*bounds)
            correct = criterion(xpert,y)

            k+=1

        return xpert
