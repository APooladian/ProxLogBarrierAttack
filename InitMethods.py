"""Loads a pretrained model, then attacks it.

   This script minimizes the distance of a misclassified image to an original,
   and enforces misclassification with a log barrier penalty.
"""
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
from foolbox.adversarial import Adversarial

from scipy.ndimage.filters import gaussian_filter

class SafetyInitialize():

    def __init__(self,model,TrainingIms):
        self.model = model
        self.Ims = TrainingIms

    def __call__(self,xinits,x,y):
        Ims = self.Ims
        for i in range(len(y)):
            yi = y[i].item()
            if xinits[i].norm() == np.inf or xinits[i].norm() == torch.tensor(np.nan).cuda():
                xinits[i] = Ims[(yi + 1) % 10].cuda()

        return xinits

class UniformInitialize():

    def __init__(self, model, tracking=False, num_model_qs = 0, max_model_qs = 5000):
        self.model = model
        self.tracking = tracking
        self.num_model_qs = num_model_qs
        self.max_model_qs = max_model_qs

    def __call__(self, x, y, criterion, max_iters=1e4, bounds=(0,1)):
        if type(x) == Adversarial:
            x = x.unperturbed
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).cuda()
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        xpert = x.clone()
        dt=0.01
        #dt = 5e-4
        Unif = torch.distributions.Uniform(-1,1)
        nbd_evals = 0
        correct = criterion(xpert,y)
        nbd_evals += 1
        k=0
        while correct.sum() > 0:

            l = correct.sum()
            USample = Unif.sample((l,*x.shape[1:])).cuda()
            xpert[correct] = x[correct] + 1.01**k*dt*USample
            xpert.clamp_(*bounds)
            correct = criterion(xpert,y)
            nbd_evals += 1
            k+=1

            if k>max_iters:
                print('Failed to initialize: maximum iterations reached')
                xpert[correct] = torch.full(x.shape[1:],np.inf).cuda()
                return xpert.squeeze_(0)
                #raise ValueError('Failed to initialize: maximum iterations reached')

        return xpert.squeeze_(0)


class GaussianInitialize():

    def __init__(self, model, tracking=False, num_model_qs = 0, max_model_qs = 5000):
        self.model = model
        self.tracking = tracking
        self.num_model_qs = num_model_qs
        self.max_model_qs = max_model_qs

    def __call__(self, x, y, criterion, max_iters=1e4, bounds=(0,1)):
        """Generates image perturbed with gaussian noise until incorrect label"""
        if type(x) == Adversarial:
            x = x.unperturbed
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).cuda()
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        xpert = x.clone()
        #dt = 5e-4 #; originally 0.01
        dt = 0.01
        correct = criterion(xpert,y)
        k=0
        while correct.sum()>0:
            l = correct.sum()
            xpert[correct] = x[correct] + (1.01)**k*dt*torch.randn(l,*xpert.shape[1:],
                                                        device=xpert.device).cuda()
            xpert.clamp_(*bounds)
            correct = criterion(xpert,y)

            k+=1
            if k>max_iters:
                print('Failed to initialize: maximum iterations reached')
                xpert[correct] = torch.full(x.shape[1:],np.inf).cuda()
                return xpert.squeeze_(0)
                #raise ValueError('Failed to initialize: maximum iterations reached')

        return xpert.squeeze_(0)


class HeatSmoothing(nn.Module):
    def __init__(self,dt=0.5,iters=1,nchannels=1):
        super().__init__()
        self.dt = dt
        self.iters = iters
        self.nchannels=nchannels
        self.L = nn.ReflectionPad2d((1,1,1,1))
        self.register_buffer('weight',torch.FloatTensor([[0, self.dt*0.125, 0], [self.dt*0.125, (1 - self.dt*0.5), self.dt*0.125], [0, self.dt*0.125, 0]]) )
        self.weight.unsqueeze_(0)
        self.weight.unsqueeze_(0)
        self.weight = self.weight.expand(self.nchannels,1,3,3)
        self.weight = self.weight.cuda()

    def update(self,u):
        u = self.L(u)
        u = F.conv2d(u,self.weight,padding=0,groups=self.nchannels)
        return u

    def forward(self,x):
        for n in range(self.iters):
            x = self.update(x)
        return x


class BlurInitialize():

    def __init__(self, model, tracking=False, num_model_qs = 0, max_model_qs = 5000):
        self.model = model
        self.tracking = tracking
        self.num_model_qs = num_model_qs
        self.max_model_qs = max_model_qs

    def __call__(self, x, y, criterion, max_iters=1e4, bounds=(0,1)):
        """blur images until until images have incorrect label"""
        if type(x) == Adversarial:
            x = x.unperturbed
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).cuda()
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        xpert = x.clone()
        #dt = 5e-4 #; originally 0.01
        dt = 0.1
        correct = criterion(xpert,y)
        k=0
        while correct.sum()>0:
            l = correct.sum()

            var = (1.01)**k*dt
            GFilter = HeatSmoothing(dt=var,iters=1,nchannels=1)
            xpert[correct] = GFilter(x[correct])

            xpert.clamp_(*bounds)
            correct = criterion(xpert,y)

            k+=1
            if k>max_iters:
                print('Failed to initialize: maximum iterations reached')
                xpert[correct] = torch.full((*x.shape[1:]),np.inf)
                return xpert.squeeze_(0)
                #raise ValueError('Failed to initialize: maximum iterations reached')

        return xpert.squeeze_(0)
