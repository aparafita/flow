"""
Abstract class for U priors and implementations for common-use priors.

These priors provide sampling functionality and the computation of the 
negative log-likelihood (nll) of a sample u.

Classes:

* `Prior`: abstract class for a prior.
* `Normal`: standard normal distribution (uni- or multivariate).
"""

import numpy as np

import torch
from torch import nn


class Prior(nn.Module):
    """Prior class to encapsule distribution prior for variable U.

    Any class that inherits from Prior needs to implement:
    ```python
    def sample(self, n):
        # Return n samples from this prior, mapped to the prior device.
        ...

    def nll(self, u):
        # Return the Negative Log-Likelihood (nll) of each sample in u.
        ...
    ```
    """

    def __init__(self, dim=1):
        """
        Args:
            dim (int): dimensionality for this prior. Defaults to 1.
        """
        super().__init__()

        self.dim = dim
        self.device = torch.device('cpu')

    def sample(self, n):
        """Sample n samples from this prior, mapped to the prior device."""
        raise NotImplementedError()

    def nll(self, u):
        """Return the negative log-likelihood of samples u."""
        raise NotImplementedError()

    # Device overrides
    def _update_device(self, device):
        """Update saved device for this prior."""
        self.device = device

    def to(self, device):
        """Override .to(device) so as to call _update_device(device)."""
        self._update_device(device)

        return super().to(device)

    def cpu(self):
        """Override .cpu so as to call .to method."""
        return self.to(torch.device('cpu'))

    def cuda(self):
        """Override .cuda so as to call .to method."""
        return self.to(torch.device('cuda', index=0))


class Normal(Prior):
    """Prior for a standard Normal distribution."""

    def sample(self, n):
        return torch.randn(n, self.dim, device=self.device)

    def nll(self, u):
        return .5 * (self.dim * np.log(2 * np.pi) + (u ** 2).sum(dim=1))