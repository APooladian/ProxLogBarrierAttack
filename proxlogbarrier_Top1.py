import argparse
import os, sys
import numpy as np
import pandas as pd
import pickle as pk
import torch
from torch import nn
from torch.autograd import grad
import time


from .prox import L0NormProx_Batch, LinfNormProx, L2NormProx_Batch, L1NormProx

def Top1Criterion(x,y,model):
    return model(x).topk(1)[1].view(-1) == y

class Attack():

    def __init__(self,model,criterion=Top1Criterion,norm=0,
                        verbose=True,**kwargs):

        super().__init__()
        self.model = model
        self.criterion = lambda x,y : criterion(x,y,model)
        self.labels = None

        self.norm = norm
        self.verbose = verbose

        ##default parameters for MNIST, Fashion-MNIST, CIFAR10
        config = {'bounds':(0,1),
                    'dt' : 0.1,
                    'alpha' : 0.1,
                    'beta' : 0.75,
                    'gamma' : 0.05,
                    'max_outer' : 30,
                    'max_inner': 30,
                    'T': 1}

        config.update(kwargs)
        self.hyperparams = config

    def __call__(self,xorig,xpert,y):
        norm = self.norm
        config = self.hyperparams
        model=self.model
        criterion=self.criterion

        bounds,dt,alpha0,beta,gamma,max_outer,max_inner,T = (
            config['bounds'], config['dt'], config['alpha'],
            config['beta'], config['gamma'], config['max_outer'],
            config['max_inner'], config['T'])

        Nb = len(y)
        ix = torch.arange(Nb,device=xorig.device)

        imshape = xorig.shape[1:]
        PerturbedImages = torch.full(xorig.shape,np.nan,device=xpert.device)

        #perturb only those that are correctly classified
        mis0 = criterion(xorig,y)
        xpert[~mis0] = xorig[~mis0]

        xold = xpert.clone()
        xbest = xpert.clone()
        diffBest = torch.full((Nb,),np.inf,device=xorig.device)

        #initial parameter calls
        dtz = dt*torch.ones(Nb).cuda()
        muz = T*torch.ones(Nb).cuda()

        xpert.requires_grad_(True)

        if norm == 0:
            proxFunc = L0NormProx_Batch()
        elif norm == 2:
            proxFunc = L2NormProx_Batch()
        elif norm == np.inf:
            proxFunc = LinfNormProx()
        elif norm == 1:
            proxFunc = L1NormProx()

        for k in range(max_outer):
            alpha = alpha0*beta**k

            diff = (xpert - xorig).view(Nb,-1).norm(self.norm,-1)
            update= diff>0

            for j in range(max_inner):
                p = model(xpert)
                pdiff = p.max(dim=-1)[0] - p[ix,y]
                s = -torch.log(pdiff).sum()
                g = grad(alpha*s,xpert)[0]

                with torch.no_grad():
                    if norm in [0,2]:
                        Nb_ = xpert[update].shape[0]
                        yPert = xpert[update].view(Nb_,-1) -dtz[update].view(-1,1) * g[update].view(Nb_,-1)
                        y_proxd = proxFunc(yPert,xorig[update].view(Nb_,-1),muz[update])
                        xpert[update] = y_proxd.view(Nb_,*imshape).clamp_(*bounds)
                    elif norm in [1,np.inf]:
                        Nb_ = xpert[update].shape[0]
                        yPert = xpert[update].view(Nb_,-1) -dtz[update].view(-1,1) * g[update].view(Nb_,-1)
                        y_proxd = proxFunc(yPert,xorig[update].view(Nb_,-1),T)
                        xpert[update] = y_proxd.view(Nb_,*imshape).clamp_(*bounds)

                with torch.no_grad():
                    c = criterion(xpert,y)
                    while c.any():
                        ## backtracking into feasible region ##
                        xpert[c] = xpert[c].clone().mul(gamma).add(1-gamma,xold[c])
                        c = criterion(xpert,y)


                ## keep track of best iterate
                diff = (xpert - xorig).view(Nb,-1).norm(self.norm,-1)
                boolDiff = diff <= diffBest
                xbest[boolDiff] = xpert[boolDiff]
                diffBest[boolDiff] = diff[boolDiff]

                xold = xpert.clone()


                if self.verbose:
                    sys.stdout.write('  [%2d outer, %4d inner] median & max distance: (%4.4f, %4.4f)\r'
                         %(k, j, diffBest.median() , diffBest.max()))

        if self.verbose:
            sys.stdout.write('\n')

        switched = ~criterion(xbest,y)
        PerturbedImages[switched] = xbest.detach()[switched]

        return PerturbedImages




