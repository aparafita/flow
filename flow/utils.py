"""
Miscellaneous utility functions.
"""

from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F


def log_sum_exp_trick(x, log_w=None, dim=-1, keepdim=False):
    """Compute log(sum(w * exp(x), dim=dim, keepdim=keepdim)) safely.

    Uses the logsumexp trick for the computation of this quantity.
    """
    if log_w is None:
        log_w = torch.zeros_like(x)

    x = x + log_w # this way we include w in the computation of M

    M = x.max(dim=dim, keepdim=True).values
    x = torch.log(torch.exp(x - M).sum(dim=dim, keepdim=True)) + M
    
    if not keepdim:
        x = x.squeeze(dim=dim)

    return x


def log_mean_exp_trick(x, dim=-1, keepdim=False):
    """Computes log(mean(exp(x), dim=dim, keepdim=keepdim)) safely.

    Uses the logsumexp trick for the computation of this quantity.
    """
    N = x.size(dim)

    return log_sum_exp_trick(x, dim=dim, keepdim=keepdim) - np.log(N)


softplus = lambda x, eps=1e-6, **kwargs: F.softplus(x, **kwargs) + eps

def softplus_inv(x, eps=1e-6, threshold=20.):
    """Compute the softplus inverse."""
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