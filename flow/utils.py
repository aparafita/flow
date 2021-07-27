"""
Miscellaneous utility functions.
"""

from functools import wraps

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch_utils import Module
from torch_utils.prob import log_sum_exp_trick, log_mean_exp_trick
from torch_utils.dec import cache, cache_only_in_eval


def prepend_cond(x, cond=None):
    """Return torch.cat([cond, x], 1), broadcasting cond if necessary.

    If cond is None, does nothing to x. Useful to avoid checking for cond
    and preprocessing it every time.
    """
    if cond is None:
        return x
    else:
        if cond.size(0) < x.size(0):
            cond = cond.repeat(x.size(0) // cond.size(0), 1)

        assert cond.size(0) == x.size(0)
        return torch.cat([cond, x], 1)


softplus = lambda x, eps=1e-6, **kwargs: F.softplus(x, **kwargs) + eps

def softplus_inv(x, eps=1e-6, threshold=20.):
    """Compute the softplus inverse."""
    x = x.clamp(0.)
    y = torch.zeros_like(x)

    idx = x < threshold
    # We deliberately ignore eps to avoid -inf
    y[idx] = torch.log(torch.exp(x[idx]) - 1)
    y[~idx] = x[~idx]

    return y

logsigmoid = lambda x, alpha=1., **kwargs: -softplus(-alpha * x, **kwargs)


@torch.no_grad()
def monotonic_increasing_bounded_bijective_search(
    f, x, *args, a=0., b=1., eps=1e-3, **kwargs
):
    """Use bounded bijective search to solve x = f(u) for u."""
    
    assert a < b
    a, b = torch.ones_like(x) * float(a), torch.ones_like(x) * float(b)

    diff = eps * 2
    while diff > eps:
        u = (a + b) / 2.
        fu = f(u, *args, **kwargs)

        lt = fu < x
        a = torch.where(lt, u, a)
        b = torch.where(lt, b, u)

        diff = (fu - x).abs().mean()

    return u


@torch.no_grad()
def monotonic_increasing_bijective_search(
    f, x, *args, a=-np.inf, b=np.inf, eps=1e-3, max_steps=100, **kwargs
):
    """Use unbounded bijective search to solve x = f(u) for u."""

    sig = lambda x, alpha: 1 / (1 + torch.exp(-alpha * x))
    logit = lambda x, alpha: -torch.log(1 / x - 1) / alpha

    assert a < b
    a, b = torch.ones_like(x) * a, torch.ones_like(x) * b
    alpha = torch.ones_like(x)
    
    i_a, i_b = sig(a, alpha), sig(b, alpha)
    
    diff = eps * 2
    n_steps = 0
    while diff >= eps:
        i_u = (i_a + i_b) / 2.
        u = logit(i_u, alpha)
        # Update alpha so that logit(i_u) has derivative 1
        # Get the original a, b, u
        a, b, u = logit(i_a, alpha), logit(i_b, alpha), logit(i_u, alpha)
        # Compute the new alpha (controlled so it doesn't go to inf)
        alpha = 1 / 1000 * (i_u - i_u ** 2)
        # alpha = alpha.clamp(.01, 10.)
        # Obtain the corresponding i_a, i_b, i_u
        i_a, i_b, i_u = sig(a, alpha), sig(b, alpha), sig(u, alpha)
        
        # Compute the image of u, and update bounds
        fu = f(u, *args, **kwargs)
            
        lt, gt = fu < x, fu > x
        i_a = torch.where(lt, i_u, i_a)
        i_b = torch.where(gt, i_u, i_b)
        
        # Can we stop early?
        diff = (fu - x).abs().max() # max to continue until we get the furthest point
        n_steps += 1

        if max_steps and n_steps >= max_steps:
            break

    return u


class MultiHeadNet(Module):
    
    def __init__(self, input_dim, output_dim, head_slices=[], use_dropout=True, use_bn=True, init=None):            
        assert (
            isinstance(head_slices, (list, tuple, set)) and 
            all(isinstance(s, slice) for s in head_slices)
        )
        
        super().__init__()
        
        self.head_slices = head_slices
        
        self.register_buffer('indexer', torch.ones(1, input_dim, dtype=bool))
        for s in head_slices:
            self.indexer[:, s] = False
            
        head_dims = [ self.indexer[:, s].size(1) for s in head_slices ]
        net_dim = input_dim - sum(head_dims)
        
        if net_dim:
            self.net = nn.Sequential(
                nn.BatchNorm1d(net_dim, affine=False),
                nn.Linear(net_dim, 100),
                *((nn.Dropout(),) if use_dropout else tuple()),
                nn.ReLU(),
                *((nn.BatchNorm1d(100, affine=False),) if use_bn else tuple()),
            )
        else:
            self.net = Parameter(torch.randn(1, 100))
                
        combinations = [ max(d, 2) for d in head_dims ]
        
        def combs(lens):
            if not lens:
                yield []
            else:
                for i in range(lens[0]):
                    for rest in combs(lens[1:]):
                        yield [i] + rest
                        
        def onehot(comb, lens):
            res = []
            
            for c, l in zip(comb, lens):
                res += [ c == i for i in range(l) ]
                
            return res
        
        self.register_buffer('combinations', torch.Tensor([
            onehot(comb, combinations)
            for comb in combs(combinations)
        ]).int())
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(100, 50),
                *((nn.Dropout(),) if use_dropout else tuple()),
                nn.ReLU(),
                *((nn.BatchNorm1d(50, affine=False),) if use_bn else tuple()),
                nn.Linear(50, output_dim)
            )
            for i in range(max(1, len(self.combinations))) # if there's no combinations,
            # leave one head to act as the default net
        ])
        
        if init is not None:
            # Initialize all weights and biases to close to 0,
            # and the last biases to init
            for p_it in [self.net.parameters()] + [h.parameters() for h in self.heads]:
                for p in p_it:
                    p.data = torch.randn_like(p) * 1e-3

            for head in self.heads:
                last_bias = head[-1].bias
                last_bias.data = init.to(last_bias.device)
                
    def _update_device(self, device):
        super()._update_device(device)
        
        for head in self.heads:
            head.device = device
        
        
    # To override:
    def forward(self, x):
        n = x.size(0)
        
        if self.combinations.numel(): # any combination at all?
            comb = (torch.cat([
                t if t.size(1) > 1 else torch.cat([1 - t, t], 1)
                for t in (x[:, s] for s in self.head_slices)
            ], 1) > .5).int()

            comb_idx = torch.arange(len(self.combinations)).view(1, -1).repeat(n, 1)[
                (comb.unsqueeze(1) == self.combinations.unsqueeze(0)).all(2)
            ]

            assert len(comb_idx.shape) == 1
        else:
            comb_idx = torch.zeros_like(x[:, 0]).int()
        
        x = x[self.indexer.repeat(n, 1)].view(n, -1) # remove comb
        x = self.net(x)
        
        results = []
        for i in torch.unique(comb_idx):
            idx = comb_idx == i
            head = self.heads[i]
            
            # It may be the case that when restricting to this comb,
            # there's only 1 or 0 samples. 
            # BatchNorm would raise an Exception in that case
            # since it's in training mode, so we'll skip this problem
            # by moving it to eval if this is the case, only for this evaluation.
            eval_bn = head.training and idx.sum(0).item() < 2
            if eval_bn:
                for layer in head:
                    if layer.__class__.__name__.startswith('BatchNorm'):
                        layer.eval()
            
            results.append((idx, head(x[idx])))
            
            if eval_bn:
                head.train() # reset BatchNorm's training status
            
        assert len({xi.size(1) for _, xi in results}) == 1
        result = torch.zeros(x.size(0), results[0][1].size(1), device=x.device)
        for idx, x in results:
            result[idx] = x
                    
        return result
    
    def warm_start(self, x):
        bn = self.net[0] # BatchNorm

        bn.running_mean.data = x.mean(0)
        bn.running_var.data = x.var(0)

        return self