"""
Miscellaneous Flows.
"""

import numpy as np
import torch
from torch import nn

from .flow import Flow
from .utils import softplus, softplus_inv, logsigmoid


class Affine(Flow):
    """Learnable Affine Flow.

    Applies weight[i] * x[i] + bias[i], 
    where weight and bias are learnable parameters.
    """

    def __init__(self, weight=None, bias=None, **kwargs):
        """
        Args:
            weight (torch.Tensor): initial value for the weight parameter. 
                If None, initialized to torch.ones(1, self.dim).
            bias (torch.Tensor): initial value for the bias parameter. 
                If None, initialized to torch.zeros(1, self.dim).
        """
        super().__init__(**kwargs)

        if weight is None:
            weight = torch.ones(1, self.dim)

        assert (weight > 0).all()
        self.log_weight = nn.Parameter(torch.log(weight))

        if bias is None:
            bias = torch.zeros(1, self.dim)

        self.bias = nn.Parameter(bias)

    def _log_det(self):
        """Used to compute _log_det for _transform."""
        return self.log_weight.sum(dim=1)

    def _h(self):
        """Compute the parameters for this flow."""
        return torch.exp(self.log_weight), self.bias

    def _transform(self, x, log_det=False, **kwargs):
        weight, bias = self._h()

        u = weight * x + bias

        if log_det:
            return u, self._log_det()
        else:
            return u

    def _invert(self, u, log_det=False, **kwargs):
        weight, bias = self._h()

        x = (u - bias) / weight

        if log_det:
            return x, -self._log_det()
        else:
            return x


class Sigmoid(Flow):
    """Sigmoid Flow."""

    def __init__(self, alpha=1., eps=1e-2, **kwargs):
        r"""
        Args:
            alpha (float): alpha parameter for the sigmoid function: 
                \(s(x, \alpha) = \frac{1}{1 + e^{-\alpha x}}\).
                Must be bigger than 0.
            eps (float): transformed values will be clamped to (eps, 1 - eps) 
                on both _transform and _invert.
        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.eps = eps

    def _log_det(self, x):
        """Return log|det J_T|, where T: x -> u."""
        return (
            np.log(self.alpha) + 
            2 * logsigmoid(x, alpha=self.alpha) +
            -self.alpha * x
        ).sum(dim=1)

    # Override methods
    def _transform(self, x, log_det=False, **kwargs):
        u = torch.sigmoid(self.alpha * x)
        u = u.clamp(self.eps, 1 - self.eps)

        if log_det:
            return u, self._log_det(x)
        else:
            return u

    def _invert(self, u, log_det=False, **kwargs):
        u = u.clamp(self.eps, 1 - self.eps)
        x = -torch.log(1 / self.alpha / u - 1)

        if log_det:
            return x, -self._log_det(x)
        else:
            return x


class Softplus(Flow):
    """Softplus Flow."""

    def __init__(self, threshold=20., eps=1e-6, **kwargs):
        """
        Args:
            threshold (float): values above this revert to a linear function. 
                Default: 20.
            eps (float): lower-bound to the softplus output.
        """
        super().__init__(**kwargs)

        assert threshold > 0 and eps > 0
        self.threshold = threshold
        self.eps = eps

    def _log_det(self, x):
        return logsigmoid(x).sum(dim=1)

    # Override methods
    def _transform(self, x, log_det=False, **kwargs):
        u = softplus(x, threshold=self.threshold, eps=self.eps)

        if log_det:
            return u, self._log_det(x)
        else:
            return u

    def _invert(self, u, log_det=False, **kwargs):
        x = softplus_inv(u, threshold=self.threshold, eps=self.eps)

        if log_det:
            return x, -self._log_det(x)
        else:
            return x 


class LogSigmoid(Flow):
    """LogSigmoid Flow, defined for numerical stability."""

    def __init__(self, alpha=1., **kwargs):
        """
        Args:
            alpha (float): alpha parameter used by the `Sigmoid`.
        """
        super().__init__(**kwargs)

        self.alpha = alpha

    def _log_det(self, x):
        """Return log|det J_T|, where T: x -> u."""

        return logsigmoid(-self.alpha * x).sum(dim=1) + np.log(self.alpha)

    # Override methods
    def _transform(self, x, log_det=False, **kwargs):
        u = logsigmoid(x, alpha=self.alpha)

        if log_det:
            return u, self._log_det(x)
        else:
            return u

    def _invert(self, u, log_det=False, **kwargs):
        x = -softplus_inv(-u) / self.alpha

        if log_det:
            return x, -self._log_det(x)
        else:
            return x
    

class LeakyReLU(Flow):
    """LeakyReLU Flow."""

    def __init__(self, negative_slope=0.01, **kwargs):
        """
        Args:
            negative_slope (float): slope used for those x < 0,
        """
        super().__init__(**kwargs)

        self.negative_slope = negative_slope

    def _log_det(self, x):
        return torch.where(
            x >= 0, 
            torch.zeros_like(x), 
            torch.ones_like(x) * np.log(self.negative_slope)
        ).sum(dim=1)


    # Override methods
    def _transform(self, x, log_det=False, **kwargs):
        u = torch.where(x >= 0, x, x * self.negative_slope)

        if log_det:
            return u, self._log_det(x)
        else:
            return u

    def _invert(self, u, log_det=False, **kwargs):
        x = torch.where(u >= 0, u, u / self.negative_slope)

        if log_det:
            return x, -self._log_det(x)
        else:
            return x


class BatchNorm(Flow):
    """Perform BatchNormalization as a Flow class.

    If not affine, just learns batch statistics to normalize the input.
    """

    @property
    def affine(self):
        return self._affine.item()
    

    def __init__(self, affine=True, momentum=.1, eps=1e-5, **kwargs):
        """
        Args:
            affine (bool): whether to learn parameters loc/scale.
            momentum (float): value used for the moving average
                of batch statistics. Must be between 0 and 1.
            eps (float): lower-bound for the scale tensor.
        """
        super().__init__(**kwargs)

        assert 0 <= momentum and momentum <= 1

        self.register_buffer('eps', torch.tensor(eps))
        self.register_buffer('momentum', torch.tensor(momentum))

        self.register_buffer('updates', torch.tensor(0))

        self.register_buffer('batch_loc', torch.zeros(1, self.dim))
        self.register_buffer('batch_scale', torch.ones(1, self.dim))

        assert isinstance(affine, bool)
        self.register_buffer('_affine', torch.tensor(affine))

        # We'll save these two parameters even if _affine is not True
        # because, otherwise, when we load the flow,
        # if affine has not the same value as the state_dict, 
        # it will raise an Exception.
        self.loc = nn.Parameter(torch.zeros(1, self.dim))
        self.log_scale = nn.Parameter(torch.zeros(1, self.dim))

    def warm_start(self, x):
        with torch.no_grad():
            self.batch_loc = x.mean(0, keepdim=True)
            self.batch_scale = x.std(0, keepdim=True) + self.eps

            self.updates.data = torch.tensor(1).to(self.device)

        return self

    def _activation(self, x=None, update=None):
        if self.training:
            assert x is not None and x.size(0) >= 2, \
                'If training BatchNorm, pass more than 1 sample.'

            bloc = x.mean(0, keepdim=True)
            bscale = x.std(0, keepdim=True) + self.eps

            # Update self.batch_loc, self.batch_scale
            with torch.no_grad():
                if self.updates.data == 0:
                    self.batch_loc.data = bloc
                    self.batch_scale.data = bscale
                else:
                    m = self.momentum
                    self.batch_loc.data = (1 - m) * self.batch_loc + m * bloc
                    self.batch_scale.data = \
                        (1 - m) * self.batch_scale + m * bscale

                self.updates += 1
        else:
            bloc, bscale = self.batch_loc, self.batch_scale

        loc, scale = self.loc, self.log_scale

        scale = torch.exp(scale) + self.eps
        # Note that batch_scale does not use activation,
        # since it is already in scale units.

        return bloc, bscale, loc, scale

    def _log_det(self, bscale):
        if self.affine:
            return (self.log_scale - torch.log(bscale)).sum(dim=1)
        else:
            return -torch.log(bscale).sum(dim=1)

    def _transform(self, x, log_det=False, **kwargs):
        bloc, bscale, loc, scale = self._activation(x)
        u = (x - bloc) / bscale 
        if self.affine:
            u = u * scale + loc
        
        if log_det:
            log_det = self._log_det(bscale)
            return u, log_det
        else:
            return u

    def _invert(self, u, log_det=False, **kwargs):
        assert not self.training, (
            'If using BatchNorm in reverse training mode, '
            'remember to call it reversed: inv_flow(BatchNorm)(dim=dim)'
        )

        bloc, bscale, loc, scale = self._activation()
        if self.affine:
            x = (u - loc) / scale * bscale + bloc
        else:
            x = u * bscale + bloc
        
        if log_det:
            log_det = -self._log_det(bscale)
            return x, log_det
        else:
            return x


class ActNorm(Affine):
    """Implementation of Activation Normalization.
    https://arxiv.org/pdf/1807.03039.pdf

    Uses Affine implementation and provides the warm_start method
    to initialize Affine so that the transformed distribution
    has location 0 and variance 1.

    Note that ActNorm expects to call warm_start.
    An assert blocks using it in any way before warm_start has been called.
    """

    def __init__(self, eps=1e-6, **kwargs):
        """
        Args:
            eps (float): lower-bound for the weight tensor.
        """
        super().__init__(**kwargs)

        self.register_buffer('eps', torch.tensor(eps))
        self.register_buffer('initialized', torch.tensor(False))

    def warm_start(self, x):
        """Warm start for ActNorm.

        Set loc and weight so that the transformed distribution
        has location 0 and variance 1.
        """

        self.log_weight.data = -torch.log(x.std(0, keepdim=True) + self.eps)
        self.bias.data = -(x * torch.exp(self.log_weight)).mean(0, keepdim=True)

        self.initialized.data = torch.tensor(True).to(self.device)

        return self

    def _h(self):
        assert self.initialized.item()

        return super()._h()


class Shuffle(Flow):
    """Perform a dimension-wise permutation."""
    
    def __init__(self, perm=None, **kwargs):
        """
        Args:
            perm (torch.Tensor): permutation to apply.
        """
        
        super().__init__(**kwargs)
        
        if perm is None:
            perm = torch.randperm(self.dim)
                
        assert perm.shape == (self.dim,)
        self.register_buffer('perm', perm)
        
    def _log_det(self, x):
        # By doing a permutation, det is always 1 or -1. 
        # Hence, log|det| is always 0.
        return torch.zeros_like(x[:, 0])
        
    def _transform(self, x, log_det=False, **kwargs):
        u = x[:, self.perm]
        
        if log_det:
            return u, self._log_det(x)
        else:
            return u
        
    def _invert(self, u, log_det=False, **kwargs):
        inv_perm = torch.argsort(self.perm)
        x = u[:, inv_perm]
        
        if log_det:
            return x, -self._log_det(x)
        else:
            return x