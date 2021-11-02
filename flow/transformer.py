"""
Implementations for Flow-transformers.

Of particular interest are:

* `Affine`: affine transformation.
* `DSF`: Deep Sigmoidal Flow.
* `NonAffine`: non-affine transformation.
"""

from functools import partial

import torch
from torch import nn, optim
import torch.nn.functional as F

from .flow import Flow
from .prior import Uniform
from .modules import LogSigmoid, LeakyReLU
from .utils import *

from .splines import *


class Affine(Flow):
    """Affine Flow.
    """

    def __init__(self, eps=1e-6, **kwargs):
        """
        Args:
            eps (float): lower-bound for scale parameter.
        """

        super().__init__(**kwargs)
        
        self.theta_dims = 2
        self.eps = eps

    def _log_abs_det(self, scale):
        return torch.log(scale).sum(dim=1)

    def _activation(self, theta):
        """Returns (loc, scale) parameters."""
        loc, scale = theta[:, ::2], theta[:, 1::2]
        scale = F.softplus(scale) + self.eps

        return loc, scale

    def _transform(self, x, loc, scale, log_abs_det=False, **kwargs):
        u = x * scale + loc

        if log_abs_det:
            return u, self._log_abs_det(scale)
        else:
            return u

    def _invert(self, u, loc, scale, log_abs_det=False, **kwargs):
        x = (u - loc) / scale

        if log_abs_det:
            return x, -self._log_abs_det(scale)
        else:
            return x

    def _theta_init(self):
        theta_init = torch.randn(self.theta_dim, device=self.device) * 1e-3
        theta_init[1::2] += softplus_inv(torch.tensor(1. - self.eps)).item()

        return theta_init
    
    
class ActNorm(Affine):
    """Implementation of Activation Normalization.
    https://arxiv.org/pdf/1807.03039.pdf

    Uses Affine implementation and provides the warm_start method
    to initialize Affine so that the transformed distribution
    has location 0 and variance 1.

    Note that ActNorm requires to be initialized with `warm_start`.
    """
    
    requires_initialization = True

    def __init__(self, eps=1e-6, **kwargs):
        """
        Args:
            eps (float): lower-bound for the weight tensor.
        """
        super().__init__(**kwargs)
        
        self.register_buffer('_distr_log_weight', torch.zeros(self.dim))
        self.register_buffer('distr_bias', torch.zeros(self.dim))

    def _warm_start(self, x, *theta, **kwargs):
        """Warm start for ActNorm.

        Set loc and weight so that the transformed distribution
        has location 0 and variance 1.
        """

        self._distr_log_weight.data = torch.log(x.std(0) + self.eps)
        self.distr_bias.data = x.mean(0)
        
        super()._warm_start(x, *theta, **kwargs)
        
    @property
    def distr_weight(self):
        return torch.exp(self._distr_log_weight)
        
    def _transform(self, x, *theta, log_abs_det=False, **kwargs):
        x = (x - self.distr_bias) / self.distr_weight
        
        res = super()._transform(x, *theta, log_abs_det=log_abs_det, **kwargs)
        
        if log_abs_det:
            u, d = res
            d = d - self._distr_log_weight.sum()
            
            return u, d
        else:
            return res
    
    def _invert(self, u, *theta, log_abs_det=False, **kwargs):
        u = u * self.distr_weight + self.distr_bias
        
        res = super()._invert(u, *theta, log_abs_det=log_abs_det, **kwargs)
        
        if log_abs_det:
            x, d = res
            d = d + self._distr_log_weight.sum()
            
            return x, d
        else:
            return res       


class _IncreasingMonotonicFlow(Flow):
    """Abstract Flow that inverts using Bijection Search, 
        specific for increasing monotonic transformers.

    Note that using this method, inversion will not be differentiable.
    Uses `flow.utils.monotonic_increasing_bijective_search`.
    """

    def __init__(self, inv_eps=1e-3, inv_steps=1000, **kwargs):
        """
        Args:
            inv_eps (float): minimum difference between f(u) and x
                allowed to stop the inversion.
            inv_steps (int): maximum number of iterations 
                before halting execution. If 0 (default) no maximum defined.
            inv_alpha (float): alpha parameter for the inversion method.
        """

        super().__init__(**kwargs)

        self.inv_eps = inv_eps
        self.inv_steps = inv_steps

    def _invert(self, u, *theta, log_abs_det=False, **kwargs):
        x = monotonic_increasing_bijective_search(
            # use _transform, but without log_abs_det
            self._transform, u, *theta, **kwargs,
            eps=self.inv_eps, max_steps=self.inv_steps
        )

        if log_abs_det:
            _, log_abs_det = self._transform(u, *theta, log_abs_det=True, **kwargs)
            return x, -log_abs_det
        else:
            return x


class _AdamInvFlow(Flow):
    """Abstract Flow that inverts using the Adam optimizer.

    Note that using this method, inversion will not be differentiable.

    **CAUTION**: for any inheriting Flows, 
    if you need to pass tensors as **kwargs to _invert, don't pass them inside
    lists or any another collection, pass them directly.
    Otherwise, _invert would run through their graph multiple times 
    and result in an Exception. See _invert for more details.
    """

    def __init__(
        self, 
        inv_lr=1e-1, inv_eps=1e-3, inv_steps=1000, inv_init=None, 
        **kwargs
    ):
        """
        Args:
            inv_lr (float): learning rate for the Adam optimizer.
                Quite high by default (1e-1) in order to make sampling fast.
                For more precision, use inv_lr=1e-3 and inv_steps >= 10000
            inv_eps (float): minimum difference between f(u) and x squared
                allowed to stop the inversion.
            inv_steps (int): maximum number of iterations 
                before halting execution. If 0 (default) no maximum defined.
            inv_init (function): function used to inicialize u.
                If None, u = torch.randn_like(x).
        """

        super().__init__(**kwargs)

        self.inv_lr = inv_lr
        self.inv_eps = inv_eps
        self.inv_steps = inv_steps
        self.inv_init = inv_init

    def _invert(self, u, *theta, log_abs_det=False, inv_init=None, **kwargs):
        # This operation will not be invertible, so use no_grad
        with torch.no_grad():
            # Avoid running twice through the graph
            u = u.clone()
            theta = tuple(theta_i.clone() for theta_i in theta)

            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwargs[k] = v.clone()

            if inv_init is not None:
                # Broadcast
                x = inv_init.repeat(u.size(0) // inv_init.size(0), 1)
                assert x.shape == u.shape
            elif self.inv_init is None:
                x = nn.Parameter(torch.randn_like(u))
            else:
                x = nn.Parameter(self.inv_init(u, *theta, **kwargs))
            
            # Howewer, we do need to enable gradients here to use the optimizer.
            with torch.enable_grad(): 
                optimizer = optim.Adam([x], lr=self.inv_lr)
                
                for _ in range(self.inv_steps):
                    loss = (
                        (u - self._transform(x, *theta, **kwargs)) ** 2
                    ).mean()
                    
                    if loss.item() < self.inv_eps:
                        break
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                x = x.data # get the data from the parameter
                
            if log_abs_det:
                _, log_abs_det = self._transform(
                    x, *theta, **kwargs, log_abs_det=True
                )
                log_abs_det = -log_abs_det # we're inverting
                
                return x, log_abs_det
            else:
                return x


class NonAffine(_AdamInvFlow):
    '''Non-affine transformer.

    https://arxiv.org/abs/1912.02762
    '''

    def __init__(self, k=16, nl=LeakyReLU, eps=1e-6, **kwargs):
        """
        Args:
            k (int): number of components of the conic combination.
            nl (class): non-linearity Flow to use in each component.
                Defaults to `flow.modules.LeakyReLU`.
            eps (float): lower-bound to strictly-positive h parameters.
        """
        super().__init__(**kwargs)
        
        self.theta_dims = 3 * k + 1
        self.k = k
        self.nl = nl()
        self.eps = eps

    def _activation(self, theta):
        """Returns (weight, loc, scale, bias) parameters."""
        assert theta.shape[1] == self.theta_dim
        
        theta = theta.view(theta.size(0), self.dim, -1)
        loc, scale, log_weight = theta[..., :-1:3], theta[..., 1:-1:3], theta[..., 2:-1:3]
        bias = theta[..., -1]

        scale = F.softplus(scale) + self.eps
        log_weight = F.log_softmax(log_weight, dim=2)

        return log_weight, loc, scale, bias

    def _transform(self, u, log_weight, loc, scale, bias, log_abs_det=False, **kwargs):
        z = u.unsqueeze(2) * scale + loc

        # We need the derivative of each dimension individually,
        # so we need to reshape to (-1, 1) first.
        shape = z.shape # save the original shape for later
        z = z.view(-1, 1)
        
        nl_res = self.nl(z, log_abs_det=log_abs_det)
        if log_abs_det:
            nl_z, log_abs_det_i = nl_res
            log_abs_det_i = log_abs_det_i.view(*shape) # restore shape

            log_abs_det_i = log_sum_exp_trick(
                log_weight + log_abs_det_i + torch.log(scale)
            ).sum(dim=1)
        else:
            nl_z = nl_res

        nl_z = nl_z.view(*shape) # restore shape
        x = (nl_z * torch.exp(log_weight)).sum(dim=2) + bias

        if log_abs_det:
            return x, log_abs_det_i
        else:
            return x

    def _theta_init(self):
        theta_init = torch.randn(self.theta_dim, device=self.device).view(self.dim, -1) * 1e-3
        # loc and bias 0, scale 1
        # weight can be random, since all components return the same result

        # theta_init[:, :-1:3] = 0 # loc
        theta_init[:, 1:-1:3] += softplus_inv(
            torch.tensor(1. - self.eps)
        ).item() # scale
        theta_init[:, 2:-1:3] += torch.randn(
            self.dim, self.theta_dims[0] // 3, device=self.device
        ) # log_weight
        
        # theta_init[:, -1] = 0 # bias

        return theta_init.flatten()


class DSF(_AdamInvFlow):
    """Deep Sigmoidal Flow.

    https://arxiv.org/abs/1804.00779
    """

    def __init__(self, k=16, eps=1e-6, alpha=1., **kwargs):
        """
        Args:
            - k (int): number of components of the conic combination.
            - eps (float): lower-bound to strictly-positive h parameters.
            - alpha (float): alpha parameter for the sigmoid. Defaults to 1.
        """

        super().__init__(**kwargs)
        
        self.theta_dims = 3 * k

        self.k = k
        self.register_buffer('eps', torch.tensor(eps))
        self.ls = LogSigmoid(dim=self.dim, alpha=alpha)

    def _activation(self, theta):
        """Returns (loc, scale, w, loc_post, scale_post) parameters."""
        theta = theta.view(theta.size(0), self.dim, -1)

        loc, scale, log_w = theta[..., ::3], theta[..., 1::3], theta[..., 2::3]
        
        scale = F.softplus(scale) + self.eps
        log_w = F.log_softmax(log_w, dim=2)
        
        return loc, scale, log_w

    def _transform(self, x, loc, scale, log_w, log_abs_det=False, **kwargs):
        # TODO: Avoid computing log_abs_det if not requested
        z = scale * x.unsqueeze(2) + loc

        # We need the derivative of each dimension individually,
        # so we need to reshape to (-1, 1) first.
        shape = z.shape # save the original shape for later
        
        z, log_abs_det_z = self.ls(z.view(-1, 1), log_abs_det=True)

        # Restore shape
        z = z.view(*shape)
        log_abs_det_z = log_abs_det_z.view(*shape)

        z2 = log_sum_exp_trick(log_w + z) # this removes the 3rd dimension
        
        # Again, we need the derivative of each dimension
        shape = z2.shape # save shape

        u, log_abs_det_u = self.ls(z2.view(-1, 1), invert=True, log_abs_det=True)
        
        # Restore shape
        u = u.view(*shape)
        log_abs_det_u = log_abs_det_u.view(*shape)

        # Finally, compute log_abs_det if required
        if log_abs_det:
            log_abs_det = (
                log_abs_det_u +
                -z2 +
                log_sum_exp_trick(
                    log_w + 
                    z + 
                    log_abs_det_z + 
                    torch.log(scale)
                )
            ).sum(dim=1)

            return u, log_abs_det
        else:
            return u

    def _theta_init(self):
        theta_init = torch.randn(self.theta_dim, device=self.device).view(self.dim, -1) * 1e-3
        # loc 0, scale 1
        # weight can be random, since all components return the same result

        # theta_init[:, ::3] = 0 # loc
        theta_init[:, 1::3] += softplus_inv(1. - self.eps) # scale
        theta_init[:, 2::3] += torch.randn(
            self.dim, self.theta_dims[0] // 3, device=self.device
        ) # log_weight

        return theta_init.flatten()

    
class RQ_Spline(Flow):
    """Neural Spline Flow, implemented for the rational quadratic case.
    
    Based on https://arxiv.org/pdf/1906.04032.pdf
    """
    
    @property
    def K(self):
        return self._K.item()
    
    @property
    def eps(self):
        return self._eps.item()
    
    def __init__(self, K=20, eps=1e-3, A=0., B=1., fA=None, fB=None, **kwargs):
        assert isinstance(K, int) and K >= 2
        if fA is None: fA = A
        if fB is None: fB = B
        assert A < B and fA < fB

        super().__init__(**kwargs)
        
        # How many parameters? K for widths and heights, K - 1 for derivatives
        self.theta_dims = 3 * K - 1
        
        self.register_buffer('_K', torch.tensor(int(K)))
        self.register_buffer('A', torch.tensor(float(A)))
        self.register_buffer('B', torch.tensor(float(B)))
        self.register_buffer('fA', torch.tensor(float(fA)))
        self.register_buffer('fB', torch.tensor(float(fB)))
        self.register_buffer('_eps', torch.tensor(eps))
        
    def _activation(self, theta, **kwargs): 
        theta = theta.view(theta.size(0), self.dim, -1)
        
        widths, heights, derivatives = theta[..., 0::3], theta[..., 1::3], theta[..., 2::3]
        
        return widths, heights, derivatives

    def _theta_init(self):
        theta = torch.randn(self.theta_dim, device=self.device).view(self.dim, -1) * 1e-3
        
        # heights = theta[0::3], which should be all 1s -> 0 pre
        # widths = theta[1::3], which should be all 1 / K -> 0 pre
        # derivatives = theta[2::3], which should be 1 -> softplus^-1(1 - self.eps)
        
        theta[..., 2::3] += softplus_inv(1. - self._eps)
        
        return theta.flatten()
    
    def _forward(self, x, widths, heights, derivatives, log_abs_det=False, invert=False):
        outputs, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=x,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=False,
            min_bin_width=self.eps,
            min_bin_height=self.eps,
            min_derivative=self.eps,
            A=self.A, B=self.B, fA=self.fA, fB=self.fB,
        )
        
        if log_abs_det:
            return outputs, logabsdet.view(outputs.size(0), -1).sum(1)
        else:
            return outputs
    
    def _transform(self, x, *theta, log_abs_det=False, **kwargs):
        return self._forward(x, *theta, log_abs_det=log_abs_det, invert=False)

    def _invert(self, x, *theta, log_abs_det=False, **kwargs):
        return self._forward(x, *theta, log_abs_det=log_abs_det, invert=True)
    

class Q_Spline(Flow):
    """Neural Spline Flow, implemented for the quadratic case.
    
    Models distributions from and to the unit hypercube [0, 1]^K. 
    Apply a Sigmoid + Affine flow before and after 
    to transform it back to the real line.

    Defaults to a Uniform prior.
    """
    
    @property
    def K(self):
        return self._K.item()
    
    def __init__(self, K=20, eps=1e-6, **kwargs):
        assert isinstance(K, int) and K >= 2
        
        super().__init__(**kwargs)
        
        # How many parameters? K for the widths (not considering 0 and 1)
        # and K + 1 for each cut output value.
        self.theta_dims = 2 * K + 1
        
        self.register_buffer('_K', torch.tensor(int(K)))
        self.register_buffer('eps', torch.tensor(eps))
        
    def _activation(self, theta, **kwargs): 
        theta = theta.view(theta.size(0), self.dim, -1)
        
        heights, widths = theta[..., 0::2], theta[..., 1::2]
        widths = torch.softmax(widths, -1)
        
        heights = torch.exp(heights) / (
            (
                torch.exp(heights[..., :-1]) + 
                torch.exp(heights[..., 1:])
            ) / 2 * widths
        ).sum(-1, keepdim=True)

        return widths.flatten(1), heights.flatten(1)

    def _theta_init(self):
        theta = torch.randn(self.theta_dim, device=self.device).view(self.dim, -1) * 1e-3
        
        # heights = theta[:, 0::2], which should be all 1s -> 0 pre
        # widths = theta[:, 1::2], which should be all 1 / K -> 0 pre
        
        return theta.flatten()
    
    def _lerp(self, a, b, x):
        return (b - a) * x + a
    
    def _log_abs_det(self, heights, bins, alpha):
        return torch.log(self._lerp(
            (heights[..., :-1] * bins).sum(-1),
            (heights[..., 1:] * bins).sum(-1),
            alpha
        )).sum(1)

    def _preprocess_theta(self, widths, heights):
        widths, heights = tuple(
            theta.view(theta.size(0), self.dim, -1)
            for theta in [widths, heights]
        )

        cuts = torch.cat([
            torch.zeros_like(widths[..., :1]), 
            torch.cumsum(widths, -1),
        ], -1)

        return widths, heights, cuts

    def _transform(self, x, widths, heights, log_abs_det=False, **kwargs): 
        x = x.clamp(self.eps / 2, 1 - self.eps / 2)

        widths, heights, cuts = self._preprocess_theta(widths, heights)

        # Transform x into u using parameters theta.
        x = x.unsqueeze(-1)
        bins = (cuts[..., :-1] <= x) & (x < cuts[..., 1:])
        
        alpha = (x.squeeze(-1) - (bins * cuts[..., :-1]).sum(-1)) / \
            (bins * widths).sum(-1)
        
        u = alpha * (widths * bins).sum(-1) * (
            .5 * alpha * (heights[..., 1:] * bins).sum(-1) +
            (1 - .5 * alpha) * (heights[..., :-1] * bins).sum(-1)
        ) + (
            ((heights[..., :-1] + heights[..., 1:]) / 2.) * 
            widths * (x >= cuts[..., 1:])
        ).sum(-1)

        u = u.clamp(self.eps / 2, 1 - self.eps / 2)
        
        if log_abs_det:
            return u, self._log_abs_det(heights, bins, alpha)
        else:
            return u

    def _invert(self, u, widths, heights, log_abs_det=False, **kwargs):
        u = u.clamp(self.eps / 2, 1 - self.eps / 2)

        widths, heights, cuts = self._preprocess_theta(widths, heights)

        # Transform u into x using parameters theta.
        u = u.unsqueeze(-1)
        
        y_cuts = torch.cumsum(
            widths * (heights[..., 1:] + heights[..., :-1]) / 2,
            -1
        )
        
        y_cuts = torch.cat([
            torch.zeros_like(y_cuts[..., :1]),
            y_cuts
        ], -1)
        
        bins = (y_cuts[..., :-1] <= u) & (u < y_cuts[..., 1:])
        
        a = .5 * (widths * (heights[..., 1:] - heights[..., :-1]) * bins).sum(-1)
        b = (heights[..., :-1] * widths * bins).sum(-1)
        c = (
            ((heights[..., :-1] + heights[..., 1:]) / 2.) * 
            widths * (u >= y_cuts[..., 1:])
        ).sum(-1) - u.squeeze(-1)
        
        # Note that a can be 0 for flat segments
        x = torch.zeros_like(u.squeeze(-1))
        idx = a == 0
        x[idx] = -c[idx] / b[idx]
        idx = ~idx
        
        disc = b ** 2 - 4 * a * c
        disc = disc.clamp_min(0.) # avoid numerical error
        sq = torch.sqrt(disc)
        
        alpha1 = (-b + sq) / (2 * a)
        alpha2 = (-b - sq) / (2 * a)
        
        alpha = torch.zeros_like(alpha1)
        alpha[idx] = torch.where(
            (0 <= alpha1) & (alpha1 <= 1), 
            alpha1, 
            alpha2
        )[idx]
        
        x[idx] = (
            alpha * (widths * bins).sum(-1) + (cuts[..., :-1] * bins).sum(-1)
        )[idx]

        x = x.clamp(self.eps / 2, 1 - self.eps / 2)
        
        if log_abs_det:
            return x, -self._log_abs_det(heights, bins, alpha)
        else:
            return x
