import argparse
import os, sys
import numpy as np
import pandas as pd
import pickle as pk
import torch
import time

from torch import nn
from torch.autograd import grad

from .prox import LinfNormProx, L2NormProx_Batch, L1NormProx

def Top5Criterion(x,y,model):
    return (model(x).topk(5)[1]==y.view(-1,1)).any(dim=-1)

class Attack():

    def __init__(self,model,criterion=Top5Criterion,norm=0,
                        verbose=True,**kwargs):

        super().__init__()
        self.model = model
        self.criterion = lambda x,y : criterion(x,y,model)
        self.labels = None

        self.norm = norm
        self.verbose = verbose

        config = {'bounds':(0,1),
                    'dt' : 0.1,
                    'alpha' : 0.1,
                    'beta' : 0.75,
                    'gamma' : 0.05,
                    'max_outer' : 30,
                    'tol':1e-6,
                    'max_inner': 30,
                    'T': 1}

        config.update(kwargs)
        self.hyperparams = config


    def __call__(self,xorig,xpert,y):
        norm = self.norm
        config = self.hyperparams
        model=self.model
        criterion=self.criterion

        bounds,dt,alpha0,beta,gamma,max_outer,tol,max_inner,T = (
            config['bounds'], config['dt'], config['alpha'],
            config['beta'], config['gamma'], config['max_outer'],
            config['tol'], config['max_inner'], config['T'])

        Nb = len(y)
        ix = torch.arange(Nb,device=xorig.device)

        imshape = xorig.shape[1:]
        PerturbedImages = torch.full(xorig.shape,np.nan,device=xpert.device)


        mis0 = criterion(xorig,y)
        xpert[~mis0] = xorig[~mis0]

        xold = xpert.clone()
        xbest = xpert.clone()
        diffBest = torch.full((Nb,),np.inf,device=xorig.device)

        #initial parameter calls
        gammaz = torch.ones(Nb).cuda()

        xpert.requires_grad_(True)

        if norm == 0:
            #default for L0 perturbations on ImageNet-1k
            T = 0.5
            proxFunc = L1NormProx()
        elif norm == 2:
            proxFunc = L2NormProx_Batch()
        elif norm == np.inf:
            #defaults for Linf perturbations on ImageNet-1k
            T = 3
            max_outer = 50
            max_inner = 50
            proxFunc = LinfNormProx()

        dtz = dt*torch.ones(Nb).cuda()
        muz = T*torch.ones(Nb).cuda()

        for k in range(max_outer):
            alpha = alpha0*beta**k

            diff = (xpert - xorig).view(Nb,-1).norm(self.norm,-1)
            update = diff>0

            for j in range(max_inner):
                p = model(xpert).softmax(dim=-1)
                p_ = p.topk(5,dim=-1)[0]
                pdiff = p_ - p[ix,y].view(Nb,-1)
                s = -torch.log(pdiff).sum()
                g = grad(alpha*s,xpert)[0]

                if norm in [0,2]:
                    with torch.no_grad():
                        Nb_ = xpert[update].shape[0]
                        yPert = xpert[update].view(Nb_,-1) -dtz[update].view(-1,1) * g[update].view(Nb_,-1)
                        y_proxd = proxFunc(yPert,xorig[update].view(Nb_,-1),muz[update].view(Nb_,-1))
                        xpert[update] = y_proxd.view(Nb_,*imshape).clamp_(*bounds)
                elif norm == np.inf:
                    with torch.no_grad():
                        Nb_ = xpert[update].shape[0]
                        yPert = xpert[update].view(Nb_,-1) -dtz[update].view(-1,1) * g[update].view(Nb_,-1)
                        y_proxd = proxFunc(yPert,xorig[update].view(Nb_,-1),T)
                        xpert[update] = y_proxd.view(Nb_,*imshape).clamp_(*bounds)

                with torch.no_grad():
                    c = criterion(xpert,y)
                    crit_counter = 0
                    while c.any():
                        xpert[c] = xpert[c].clone().mul(gamma).add(1-gamma,xold[c])
                        c = criterion(xpert,y)


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




