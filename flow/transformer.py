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

from .flow import Transformer
from .prior import Uniform
from .modules import LogSigmoid, LeakyReLU
from .utils import *


class Affine(Transformer):
    """Affine Transformer.
    """

    def __init__(self, eps=1e-6, **kwargs):
        """
        Args:
            eps (float): lower-bound for scale parameter.
        """

        _h_dim = 2
        h_dim = kwargs.pop('h_dim', _h_dim)
        assert h_dim == _h_dim, f'Received h_dim={h_dim} but expected {_h_dim}'

        super().__init__(h_dim=h_dim, **kwargs)

        self.eps = eps

    def _log_det(self, scale):
        return torch.log(scale).sum(dim=1)

    def _activation(self, h):
        """Returns (loc, scale) parameters."""
        assert not h.size(1) % self.h_dim

        loc, scale = h[:, ::2], h[:, 1::2]
        scale = F.softplus(scale) + self.eps

        return loc, scale

    def _transform(self, x, *h, log_det=False, **kwargs):
        loc, scale = h
        u = x * scale + loc

        if log_det:
            return u, self._log_det(scale)
        else:
            return u

    def _invert(self, u, *h, log_det=False, **kwargs):
        loc, scale = h
        x = (u - loc) / scale

        if log_det:
            return x, -self._log_det(scale)
        else:
            return x

    def _h_init(self):
        h_init = torch.zeros(self.dim * self.h_dim, device=self.device)
        h_init[1::2] = softplus_inv(torch.tensor(1. - self.eps)).item()

        return h_init


class _IncreasingMonotonicTransformer(Transformer):
    """Abstract Transformer that inverts using Bijection Search, 
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

    def _invert(self, u, *h, log_det=False, **kwargs):
        x = monotonic_increasing_bijective_search(
            # use _transform, but without log_det
            self._transform, u, *h, **kwargs,
            eps=self.inv_eps, max_steps=self.inv_steps
        )

        if log_det:
            _, log_det = self._transform(u, *h, log_det=True, **kwargs)
            return x, -log_det
        else:
            return x


class _AdamInvTransformer(Transformer):
    """Abstract Transformer that inverts using the Adam optimizer.

    Note that using this method, inversion will not be differentiable.

    **CAUTION**: for any inheriting Transformers, 
    if you need to pass tensors as **kwargs to _invert, don't pass them inside
    lists or any another collection, pass them directly.
    Otherwise, _invert would run through their graph multiple times 
    and result in an Exception. See _invert for more details.
    """

    def __init__(
        self, 
        inv_lr=1e-1, inv_eps=1e-3, inv_steps=1000, 
        inv_init=None, **kwargs
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

        super().__init__(reversed=reversed, **kwargs)

        self.inv_lr = inv_lr
        self.inv_eps = inv_eps
        self.inv_steps = inv_steps
        self.inv_init = inv_init

    def _invert(self, u, *h, log_det=False, inv_init=None, **kwargs):
        # _invert should be called inside a torch.no_grad(), 
        # since this operation will not be invertible
        with torch.no_grad():
            # Avoid running twice through the graph
            u = u.clone()
            h = tuple(hi.clone() for hi in h)

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
                x = nn.Parameter(self.inv_init(u, *h, **kwargs))
            
            # Howewer, we do need to enable gradients here to use the optimizer.
            with torch.enable_grad(): 
                optimizer = optim.Adam([x], lr=self.inv_lr)
                
                for _ in range(self.inv_steps):
                    loss = (
                        (u - self._transform(x, *h, **kwargs)) ** 2
                    ).mean()
                    
                    if loss.item() < self.inv_eps:
                        break
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                x = x.data # get the data from the parameter
                
            if log_det:
                _, log_det = self._transform(
                    x, *h, **kwargs, log_det=True
                )
                log_det = -log_det # we're inverting
                
                return x, log_det
            else:
                return x


class NonAffine(_AdamInvTransformer):
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

        _h_dim = 3 * k + 1
        h_dim = kwargs.pop('h_dim', _h_dim)
        assert h_dim == _h_dim, f'Received h_dim={h_dim} but expected {_h_dim}'

        super().__init__(h_dim=h_dim, **kwargs)

        self.k = k
        self.nl = nl()
        self.eps = eps

    def _activation(self, h):
        """Returns (weight, loc, scale, bias) parameters."""
        assert not h.size(1) % self.h_dim

        h = h.view(h.size(0), -1, self.h_dim)
        loc, scale, log_weight = h[..., :-1:3], h[..., 1:-1:3], h[..., 2:-1:3]
        bias = h[..., -1]

        scale = F.softplus(scale) + self.eps
        log_weight = F.log_softmax(log_weight, dim=2)

        return log_weight, loc, scale, bias

    def _transform(self, u, *h, log_det=False, **kwargs):
        log_weight, loc, scale, bias = h
        z = u.unsqueeze(2) * scale + loc

        # We need the derivative of each dimension individually,
        # so we need to reshape to (-1, 1) first.
        shape = z.shape # save the original shape for later
        z = z.view(-1, 1)
        
        nl_res = self.nl(z, log_det=log_det)
        if log_det:
            nl_z, log_det_i = nl_res
            log_det_i = log_det_i.view(*shape) # restore shape

            log_det_i = log_sum_exp_trick(
                log_weight + log_det_i + torch.log(scale)
            ).sum(dim=1)
        else:
            nl_z = nl_res

        nl_z = nl_z.view(*shape) # restore shape
        x = (nl_z * torch.exp(log_weight)).sum(dim=2) + bias

        if log_det:
            return x, log_det_i
        else:
            return x

    def _h_init(self):
        h_init = torch.zeros(self.dim, self.h_dim, device=self.device)
        # loc and bias 0, scale 1
        # weight can be random, since all components return the same result

        # h_init[:, :-1:3] = 0 # loc
        h_init[:, 1:-1:3] = softplus_inv(
            torch.tensor(1. - self.eps)
        ).item() # scale
        h_init[:, 2:-1:3] = torch.randn(
            self.dim, self.h_dim // 3, device=self.device
        ) # log_weight
        
        h_init[:, -1] = 0 # bias

        return h_init.flatten()


class DSF(_AdamInvTransformer):
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

        _h_dim = 3 * k
        h_dim = kwargs.pop('h_dim', _h_dim)
        assert h_dim == _h_dim, f'Received h_dim={h_dim} but expected {_h_dim}'

        super().__init__(h_dim=h_dim, **kwargs)

        self.k = k
        self.eps = eps
        self.ls = LogSigmoid(dim=self.dim, alpha=alpha, eps=eps)

    def _activation(self, h):
        """Returns (loc, scale, w, loc_post, scale_post) parameters."""
        assert not h.size(1) % self.h_dim, (h.size(1), self.h_dim)

        h = h.view(h.size(0), -1, self.h_dim)

        loc, scale, log_w = h[..., ::3], h[..., 1::3], h[..., 2::3]
        
        scale = F.softplus(scale) + self.eps
        log_w = F.log_softmax(log_w, dim=2)
        
        return loc, scale, log_w

    def _transform(self, x, *h, log_det=False, **kwargs):
        # TODO: Avoid computing log_det if not requested
        loc, scale, log_w = h

        z = scale * x.unsqueeze(2) + loc

        # We need the derivative of each dimension individually,
        # so we need to reshape to (-1, 1) first.
        shape = z.shape # save the original shape for later
        
        z, log_det_z = self.ls(z.view(-1, 1), log_det=True)

        # Restore shape
        z = z.view(*shape)
        log_det_z = log_det_z.view(*shape)

        z2 = log_sum_exp_trick(log_w + z) # this removes the 3rd dimension
        
        # Again, we need the derivative of each dimension
        shape = z2.shape # save shape

        u, log_det_u = self.ls(z2.view(-1, 1), invert=True, log_det=True)
        
        # Restore shape
        u = u.view(*shape)
        log_det_u = log_det_u.view(*shape)

        # Finally, compute log_det if required
        if log_det:
            log_det = (
                log_det_u +
                -z2 +
                log_sum_exp_trick(
                    log_w + 
                    z + 
                    log_det_z + 
                    torch.log(scale)
                )
            ).sum(dim=1)

            return u, log_det
        else:
            return u

    def _h_init(self):
        h_init = torch.zeros(self.dim, self.h_dim, device=self.device)
        # loc 0, scale 1
        # weight can be random, since all components return the same result

        # h_init[:, ::3] = 0 # loc
        h_init[:, 1::3] = softplus_inv(
            torch.tensor(1. - self.eps)
        ).item() # scale
        h_init[:, 2::3] = torch.randn(
            self.dim, self.h_dim // 3, device=self.device
        ) # log_weight

        return h_init.flatten()


'''
class RQ_Spline(Transformer):
    """Neural Spline Flow, implemented for the rational quadratic case.
    
    Based on https://arxiv.org/pdf/1906.04032.pdf
    """
    
    @property
    def K(self):
        return self._K.item()
    
    @property
    def width(self):
        return self.B - self.A
    
    def __init__(self, K=20, eps=1e-3, A=0., B=1., **kwargs):
        assert isinstance(K, int) and K >= 2
        assert A < B

        dim = kwargs.pop('dim', 1)
        prior = kwargs.pop('prior', Uniform(dim=dim))
        
        # How many parameters? K for widths and heights, K - 1 for derivatives
        h_dim = 3 * K - 1
        super().__init__(h_dim=h_dim, prior=prior, dim=dim, **kwargs)
        
        self.register_buffer('_K', torch.tensor(int(K)))
        self.register_buffer('A', torch.tensor(float(A)))
        self.register_buffer('B', torch.tensor(float(B)))
        self.register_buffer('eps', torch.tensor(eps))
        
    def _activation(self, h, **kwargs): 
        h = h.view(h.size(0), -1, self.h_dim)
        
        widths, heights, derivatives = h[..., 0::3], h[..., 1::3], h[..., 2::3]
        
        widths = (self.eps + (1 - self.eps * self.K) * torch.softmax(widths, -1)) * self.width
        heights = (self.eps + (1 - self.eps * self.K) * torch.softmax(heights, -1)) * self.width
        derivatives = torch.nn.functional.softplus(derivatives) + self.eps
        
        xknots = torch.cat(
            [torch.zeros_like(widths[..., :1]), torch.cumsum(widths, -1)], -1
        ) + self.A
        yknots = torch.cat(
            [torch.zeros_like(heights[..., :1]), torch.cumsum(heights, -1)], -1
        ) + self.A
        
        derivatives = torch.cat([
            torch.ones_like(derivatives[..., :1]),
            derivatives,
            torch.ones_like(derivatives[..., :1])
        ], -1)
        
        s = (yknots[..., 1:] - yknots[..., :-1]) / (xknots[..., 1:] - xknots[..., :-1])
        
        return widths, heights, derivatives, xknots, yknots, s

    def _h_init(self):
        h = torch.randn(self.dim, self.h_dim, device=self.device) * 1e-1
        
        # heights = h[0::3], which should be all 1s -> 0 pre
        # widths = h[1::3], which should be all 1 / K -> 0 pre
        # derivatives = h[2::3], which should be 1 -> softplus^-1(1 - self.eps)
        
        h[..., 2::3] += softplus_inv(1. - self.eps)
        
        return h.flatten()
    
    def _transform(self, x, widths, heights, derivatives, xknots, yknots, s, log_det=False, **kwargs):
        # In/out of window (A, B)
        idx = (self.A <= x) & (x < self.B)
        u = torch.zeros_like(x)
        u[~idx] = x[~idx]

        # Transform x into u using parameters h
        bins = ((xknots[..., :-1] <= x.unsqueeze(-1)) & (x.unsqueeze(-1) < xknots[..., 1:])).float()

        xk, xk1 = (xknots[..., :-1] * bins).sum(-1), (xknots[..., 1:] * bins).sum(-1)
        yk, yk1 = (yknots[..., :-1] * bins).sum(-1), (yknots[..., 1:] * bins).sum(-1)
        dk, dk1 = (derivatives[..., :-1] * bins).sum(-1), (derivatives[..., 1:] * bins).sum(-1)
        sk = (s * bins).sum(-1)

        # Note that we can get to nans if x was outside bounds.
        # Filter all these terms to idx:
        x = x[idx]
        xk = xk[idx]
        xk1 = xk1[idx]
        yk = yk[idx]
        yk1 = yk1[idx]
        dk = dk[idx]
        dk1 = dk1[idx]
        sk = sk[idx]
        
        e = ((x - xk) / (xk1 - xk))

        u[idx] = (yk + (
            (yk1 - yk) * (sk * e ** 2 + dk * e * (1 - e))
        ) / (
            sk + (dk1 + dk - 2 * sk) * e * (1 - e)
        ))
        
        if log_det:
            log_det = torch.zeros_like(u)
            log_det[idx] = (
                (
                    2 * torch.log(sk) + torch.log(
                        dk1 * e ** 2 + 2 * sk * e * (1 - e) + dk * (1 - e) ** 2
                    )
                ) - (
                    2 * torch.log(sk + (dk1 + dk - 2 * sk) * e * (1 - e))
                )
            )
            log_det = log_det.sum(1)
            assert log_det.shape == (u.size(0),)
            
            return u, log_det
        else:
            return u

    def _invert(self, u, widths, heights, derivatives, xknots, yknots, s, log_det=False, **kwargs):
        # In/out of window (-B, B)
        idx = (self.A <= u) & (u < self.B)
        x = torch.zeros_like(u)
        x[~idx] = u[~idx]

        # Transform x into u using parameters h
        bins = ((yknots[..., :-1] <= u.unsqueeze(-1)) & (u.unsqueeze(-1) < yknots[..., 1:])).float()

        xk, xk1 = (xknots[..., :-1] * bins).sum(-1), (xknots[..., 1:] * bins).sum(-1)
        yk, yk1 = (yknots[..., :-1] * bins).sum(-1), (yknots[..., 1:] * bins).sum(-1)
        dk, dk1 = (derivatives[..., :-1] * bins).sum(-1), (derivatives[..., 1:] * bins).sum(-1)
        sk = (s * bins).sum(-1)
        
        # Note that we can get to nans if x was outside bounds.
        # Filter all these terms to idx:
        u = u[idx]
        xk = xk[idx]
        xk1 = xk1[idx]
        yk = yk[idx]
        yk1 = yk1[idx]
        dk = dk[idx]
        dk1 = dk1[idx]
        sk = sk[idx]

        a = (yk1 - yk) * (sk - dk) + (u - yk) * (dk1 + dk - 2 * sk)
        b = (yk1 - yk) * dk - (u - yk) * (dk1 + dk - 2 * sk)
        c = -sk * (u - yk)

        e = 2 * c / (-b - torch.sqrt(b ** 2 - 4 * a * c))
        x[idx] = (e * (xk1 - xk) + xk)
        
        if log_det:
            log_det = torch.zeros_like(x)
            log_det[idx] = (
                (
                    2 * torch.log(sk) + torch.log(
                        dk1 * e ** 2 + 2 * sk * e * (1 - e) + dk * (1 - e) ** 2
                    )
                ) - (
                    2 * torch.log(sk + (dk1 + dk - 2 * sk) * e * (1 - e))
                )
            )
            log_det = log_det.sum(1)
            assert log_det.shape == (x.size(0),)
            
            return x, -log_det
        else:
            return x
'''

class RQ_Spline(Transformer):
    """Neural Spline Flow, implemented for the rational quadratic case.
    
    Based on https://arxiv.org/pdf/1906.04032.pdf
    """
    
    @property
    def K(self):
        return self._K.item()
    
    @property
    def width(self):
        return self.B - self.A
    
    @property
    def f_width(self):
        return self.fB - self.fA
    
    def __init__(self, K=20, eps=1e-3, A=0., B=1., fA=None, fB=None, **kwargs):
        assert isinstance(K, int) and K >= 2
        assert A < B
        
        if fA is None: fA = A
        if fB is None: fB = B

        dim = kwargs.pop('dim', 1)
        prior = kwargs.pop('prior', Uniform(dim=dim))
        
        # How many parameters? K for widths and heights, K - 1 for derivatives
        h_dim = 3 * K + 1
        super().__init__(h_dim=h_dim, prior=prior, dim=dim, **kwargs)
        
        self.register_buffer('_K', torch.tensor(int(K)))
        self.register_buffer('A', torch.tensor(float(A)))
        self.register_buffer('B', torch.tensor(float(B)))
        self.register_buffer('fA', torch.tensor(float(fA)))
        self.register_buffer('fB', torch.tensor(float(fB)))
        self.register_buffer('eps', torch.tensor(eps))
        
    def _activation(self, h, **kwargs): 
        h = h.view(h.size(0), -1, self.h_dim)
        
        derivatives, widths, heights = h[..., 0::3], h[..., 1::3], h[..., 2::3]
        
        derivatives = torch.nn.functional.softplus(derivatives) + self.eps
        widths = (self.eps + (1 - self.eps * self.K) * torch.softmax(widths, -1)) * self.width
        heights = (self.eps + (1 - self.eps * self.K) * torch.softmax(heights, -1)) * self.f_width
        
        xknots = torch.cat(
            [torch.zeros_like(widths[..., :1]), torch.cumsum(widths, -1)], -1
        ) + self.A
        
        yknots = torch.cat(
            [torch.zeros_like(heights[..., :1]), torch.cumsum(heights, -1)], -1
        ) + self.fA
        
        s = (yknots[..., 1:] - yknots[..., :-1]) / (xknots[..., 1:] - xknots[..., :-1])
        
        return widths, heights, derivatives, xknots, yknots, s

    def _h_init(self):
        h = torch.randn(self.dim, self.h_dim, device=self.device) * 1e-1
        
        # heights = h[0::3], which should be all 1s -> 0 pre
        # widths = h[1::3], which should be all 1 / K -> 0 pre
        # derivatives = h[2::3], which should be 1 -> softplus^-1(1 - self.eps)
        
        h[..., 0::3] += softplus_inv(1. - self.eps)
        
        return h.flatten()
    
    def _transform(self, x, widths, heights, derivatives, xknots, yknots, s, log_det=False, **kwargs):
        # In/out of window (A, B)
        idx = (self.A <= x) & (x < self.B)
        u = torch.zeros_like(x)
        u[x < self.A] = self.fA + (x[x < self.A] - self.A) * derivatives[..., 0][x < self.A]
        u[x > self.B] = self.fB + (x[x > self.B] + self.B) * derivatives[..., -1][x > self.B]

        # Transform x into u using parameters h
        bins = ((xknots[..., :-1] <= x.unsqueeze(-1)) & (x.unsqueeze(-1) < xknots[..., 1:])).float()

        xk, xk1 = (xknots[..., :-1] * bins).sum(-1), (xknots[..., 1:] * bins).sum(-1)
        yk, yk1 = (yknots[..., :-1] * bins).sum(-1), (yknots[..., 1:] * bins).sum(-1)
        dk, dk1 = (derivatives[..., :-1] * bins).sum(-1), (derivatives[..., 1:] * bins).sum(-1)
        sk = (s * bins).sum(-1)

        # Note that we can get to nans if x was outside bounds.
        # Filter all these terms to idx:
        x = x[idx]
        xk = xk[idx]
        xk1 = xk1[idx]
        yk = yk[idx]
        yk1 = yk1[idx]
        dk = dk[idx]
        dk1 = dk1[idx]
        sk = sk[idx]
        
        e = ((x - xk) / (xk1 - xk))

        u[idx] = (yk + (
            (yk1 - yk) * (sk * e ** 2 + dk * e * (1 - e))
        ) / (
            sk + (dk1 + dk - 2 * sk) * e * (1 - e)
        ))
        
        if log_det:
            log_det = torch.zeros_like(u)
            
            log_det[u < self.fA] = torch.log(derivatives[..., 0][u < self.fA])
            log_det[u > self.fB] = torch.log(derivatives[..., -1][u > self.fB])
            
            log_det[idx] = (
                (
                    2 * torch.log(sk) + torch.log(
                        dk1 * e ** 2 + 2 * sk * e * (1 - e) + dk * (1 - e) ** 2
                    )
                ) - (
                    2 * torch.log(sk + (dk1 + dk - 2 * sk) * e * (1 - e))
                )
            )
            log_det = log_det.sum(1)
            assert log_det.shape == (u.size(0),)
            
            return u, log_det
        else:
            return u

    def _invert(self, u, widths, heights, derivatives, xknots, yknots, s, log_det=False, **kwargs):
        # In/out of window (-B, B)
        idx = (self.A <= u) & (u < self.B)
        x = torch.zeros_like(u)
        x[u < self.fA] = self.A + (u[u < self.fA] - self.fA) / derivatives[..., 0][u < self.fA]
        x[u > self.fB] = self.B + (u[u > self.fB] + self.fB) / derivatives[..., -1][u > self.fB]

        # Transform x into u using parameters h
        bins = ((yknots[..., :-1] <= u.unsqueeze(-1)) & (u.unsqueeze(-1) < yknots[..., 1:])).float()

        xk, xk1 = (xknots[..., :-1] * bins).sum(-1), (xknots[..., 1:] * bins).sum(-1)
        yk, yk1 = (yknots[..., :-1] * bins).sum(-1), (yknots[..., 1:] * bins).sum(-1)
        dk, dk1 = (derivatives[..., :-1] * bins).sum(-1), (derivatives[..., 1:] * bins).sum(-1)
        sk = (s * bins).sum(-1)
        
        # Note that we can get to nans if x was outside bounds.
        # Filter all these terms to idx:
        u = u[idx]
        xk = xk[idx]
        xk1 = xk1[idx]
        yk = yk[idx]
        yk1 = yk1[idx]
        dk = dk[idx]
        dk1 = dk1[idx]
        sk = sk[idx]

        a = (yk1 - yk) * (sk - dk) + (u - yk) * (dk1 + dk - 2 * sk)
        b = (yk1 - yk) * dk - (u - yk) * (dk1 + dk - 2 * sk)
        c = -sk * (u - yk)

        e = 2 * c / (-b - torch.sqrt(b ** 2 - 4 * a * c))
        x[idx] = (e * (xk1 - xk) + xk)
        
        if log_det:
            log_det = torch.zeros_like(x)
            
            log_det[u < self.fA] = -torch.log(derivatives[..., 0][u < self.fA])
            log_det[u > self.fB] = -torch.log(derivatives[..., -1][u > self.fB])
            
            log_det[idx] = (
                (
                    2 * torch.log(sk) + torch.log(
                        dk1 * e ** 2 + 2 * sk * e * (1 - e) + dk * (1 - e) ** 2
                    )
                ) - (
                    2 * torch.log(sk + (dk1 + dk - 2 * sk) * e * (1 - e))
                )
            )
            log_det = log_det.sum(1)
            assert log_det.shape == (x.size(0),)
            
            return x, -log_det
        else:
            return x


class Q_Spline(Transformer):
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

        prior = kwargs.pop('prior', Uniform(dim=kwargs.get('dim', 1)))
        
        # How many parameters? K for the widths (not considering 0 and 1)
        # and K + 1 for each cut output value.
        h_dim = 2 * K + 1
        super().__init__(h_dim=h_dim, prior=prior, **kwargs)
        
        self.register_buffer('_K', torch.tensor(int(K)))
        self.register_buffer('eps', torch.tensor(eps))
        
    def _activation(self, h, **kwargs): 
        h = h.view(h.size(0), self.dim, self.h_dim)
        
        heights, widths = h[..., 0::2], h[..., 1::2]
        widths = torch.softmax(widths, -1)
        
        heights = torch.exp(heights) / (
            (
                torch.exp(heights[..., :-1]) + 
                torch.exp(heights[..., 1:])
            ) / 2 * widths
        ).sum(-1, keepdim=True)

        return widths.flatten(1), heights.flatten(1)

    def _h_init(self):
        h = torch.randn(self.dim, self.h_dim, device=self.device) * 1e-1
        
        # heights = h[0::2], which should be all 1s -> 0 pre
        # widths = h[1::2], which should be all 1 / K -> 0 pre
        
        return h.flatten()
    
    def _lerp(self, a, b, x):
        return (b - a) * x + a
    
    def _log_det(self, heights, bins, alpha):
        return torch.log(self._lerp(
            (heights[..., :-1] * bins).sum(-1),
            (heights[..., 1:] * bins).sum(-1),
            alpha
        )).sum(1)

    def _preprocess_h(self, widths, heights):
        widths, heights = tuple(
            h.view(h.size(0), self.dim, -1) 
            for h in [widths, heights]
        )

        cuts = torch.cat([
            torch.zeros_like(widths[..., :1]), 
            torch.cumsum(widths, -1),
        ], -1)

        return widths, heights, cuts

    def _transform(self, x, widths, heights, log_det=False, **kwargs): 
        x = x.clamp(self.eps / 2, 1 - self.eps / 2)

        widths, heights, cuts = self._preprocess_h(widths, heights)

        # Transform x into u using parameters h.
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
        
        if log_det:
            return u, self._log_det(heights, bins, alpha)
        else:
            return u

    def _invert(self, u, widths, heights, log_det=False, **kwargs):
        u = u.clamp(self.eps / 2, 1 - self.eps / 2)

        widths, heights, cuts = self._preprocess_h(widths, heights)

        # Transform u into x using parameters h.
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
        
        if log_det:
            return x, -self._log_det(heights, bins, alpha)
        else:
            return x