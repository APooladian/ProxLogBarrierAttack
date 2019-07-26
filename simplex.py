import torch 
from torch.autograd import Function
import torch.nn as nn

def prod(l):
    p = 1
    for x in l:
        p*=x
    return p

def _proj(x, wp, dev, dim):
    """Project x onto the same non-zero subspace in which wp lives"""
    # TODO: safety checks
    p, N = x.shape

    # mask of non zero entries--this tells us where to project
    mask = (wp>0)

    # dimension indices
    i0 = torch.arange(p,dtype=torch.long, device=dev)
    i1 = torch.arange(N,dtype=torch.long, device=dev)

    # x0 is the origin
    imax = mask.argmax(dim)
    x0 = x[i0,imax]
    x0ex = x0.unsqueeze(1).expand(p,N)

    mask[i0,imax] = 0
    i0ex = i0.unsqueeze(1).expand(p,N)
    i1ex = i1.unsqueeze(0).expand(p,N)

    xp = torch.zeros(p,N, device=dev)
    i0m, i1m = i0ex[mask], i1ex[mask]
    xp[i0m,i1m] = x[i0m, i1m] - x0ex[i0m, i1m]
    xp[i0,imax] = -xp.sum(-1)
    xp.div_(2)

    return xp

def indices(*shape, device=None):
    """Implement NumPy's indices function in PyTorch"""

    ix = list()
    for i, s in enumerate(shape):
        e = torch.ones(len(shape), dtype=torch.int, device=device)
        e[i] = s
        e = e.tolist()

        ix_ = torch.arange(s, device=device,dtype=torch.long).view(e).expand(shape)
        ix.append(ix_)

    return ix

class simplexproj(Function):

    @staticmethod
    def forward(ctx, w, z, dim):
        """Orthogonal projection onto a simplex

        Args:
            w (torch.Tensor): tensor to project
            z (scalar): level of simplex
            dim (int): dimension to act on

        Returns:
            wproj (torch.Tensor): projected tensor
            wcomplement (torch.Tensor): Perpendicular complement of wproj
        """
        dev = w.device

        if dim==-1:
            dim = len(w.shape)-1
        ctx.dim=dim

        n = w.shape[dim]
        u = w.sort(descending=True, dim=dim)[0]
        cssw =  u.cumsum(dim)

        sz = [1 if not i==dim else n for i in range(len(w.shape))]
        ix = torch.arange(1,n+1,device=dev).view(sz)
        d = (z-cssw)/ix.float() + u
        rho = (d >0).sum(dim)

        sh = list(w.shape)
        sh.pop(dim)
        ix = indices(*sh, device=dev)
        ix.insert(dim, rho-1)

        s = cssw[ix]
        l = (z-s)/rho.float()
        l.unsqueeze_(dim)

        wl = w+l
        wp = wl.clamp(min=0)
        wc = w - wp

        ctx.save_for_backward(wp)

        return wp, wc


    @staticmethod
    def backward(ctx, grad_op, grad_oc):
        wp, = ctx.saved_tensors
        dim = ctx.dim
        dev = grad_op.device

        sh = wp.shape
        p = prod(wp.shape[0:-1])
        N = wp.shape[-1]

        wp = wp.view(p,N)
        grad_wp = _proj(grad_op.view(p,N), wp, dev, dim).view(sh)
        grad_wc =  grad_oc - _proj(grad_oc.view(p,N), wp, dev, dim).view(sh)

        return grad_wp+grad_wc, None, None

class SimplexProj(nn.Module):
    def __init__(self, z=1, dim=-1):
        super().__init__()
        self.proj = simplexproj.apply
        self.z = z
        self.dim = dim
    
    def forward(self, x):
        y = self.proj(x, self.z, self.dim)

        return y


class L1BallProj(nn.Module):

    def __init__(self, z=1, dim=-1):
        super().__init__()
        self.proj = SimplexProj(z=z, dim=dim)

    def forward(self,x):
        s = x.sign()
        x = x.abs()

        xp, xc = self.proj(x)

        return xp*s, xc*s

