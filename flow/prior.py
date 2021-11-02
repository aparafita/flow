"""
Abstract class for U priors and implementations for common-use priors.

These priors provide sampling functionality and the computation of the 
negative log-likelihood (nll) of a sample u.

Classes:

* `Prior`: abstract class for a prior.
* `Uniform`: Uniform(0, 1) distribution.
* `Normal`: standard normal distribution (uni- or multivariate).
* `Exponential`: Exponential(1) distribution.
"""

import numpy as np

import torch
from torch import nn

from .utils import Module


class Prior(Module):
    """Prior class to encapsulate the distribution prior of a variable U.

    Any class that inherits from Prior needs to implement:
    ```python
    def sample(self, n):
        # Return n samples from this prior, mapped to the prior device.
        ...

    def loglk(self, u):
        # Return the log-likelihood of each sample in u.
        ...
    ```
    
    Additionally, a class attribute `discrete` must be defined 
    indicating whether this Prior is discrete (True) or continuous (False).
    Defaults to False.
    """
    
    discrete = False

    def __init__(self, dim=1):
        """
        Args:
            dim (int): dimensionality for this prior. Defaults to 1.
        """
        super().__init__()

        self.dim = dim

    def sample(self, n):
        """Return n samples from this prior, mapped to the prior device."""
        raise NotImplementedError()
        
    def loglk(self, u):
        """Return the log-likelihood of each sample in u."""
        raise NotImplementedError()

    def nll(self, u):
        """Return the negative log-likelihood of samples u."""
        return -self.loglk(u)
    
    def __repr__(self):
        return f'{self.__class__.__name__}Prior({self.dim})'


class Uniform(Prior):
    """Prior for a Uniform(0, 1) distribution."""
    
    def __init__(self, *args, eps=1e-9, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.register_buffer('eps', torch.tensor(eps))

    def sample(self, n):
        return torch.rand(n, self.dim, device=self.device) * (1 - self.eps) + self.eps / 2

    def loglk(self, u):
        return -torch.full_like(u[:, 0], torch.log(1 - self.eps))


class Normal(Prior):
    """Prior for a standard Normal distribution."""

    def sample(self, n):
        return torch.randn(n, self.dim, device=self.device)

    def loglk(self, u):
        return -.5 * (self.dim * np.log(2 * np.pi) + (u ** 2).sum(dim=1))
    
    
class Exponential(Prior):
    """Prior for a Exponential(1) distribution."""
    
    def __init__(self, *args, eps=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.register_buffer('eps', torch.tensor(eps))
    
    def sample(self, n):
        u = torch.rand(n, 1, device=self.device) * (1 - self.eps) + self.eps / 2
        return -torch.log(u)
    
    def loglk(self, u):
        return (-torch.log(1 - self.eps) - u).sum(1)