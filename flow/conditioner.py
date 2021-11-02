"""
Implementations for Flow-Conditioners.

Conditioners implemented in this class:

* `AutoregressiveNaive`: conditioner defined by a separate network 
    for each dimension. Only use for one-dimensional flows.
* `MADE`: Masked Autoregressive flow for Density Estimation.
* `CouplingLayers`: CouplingLayers Conditioner.
"""

import numpy as np

import torch
from torch import nn

import torch.nn.functional as F

from .flow import Flow, Conditioner
from .modules import Shuffle
from .utils import Module, MaskedLinear, AdjacencyMaskedNet, topological_order, prepend_cond


class Constant(Conditioner):
    
    def __init__(self, trnf, *args, **kwargs):
        super().__init__(trnf, *args, **kwargs)
        
        self._theta_param = nn.Parameter(torch.randn(trnf.theta_dim))
        
        init = self.trnf._theta_init()
        if init is not None:
            self._theta_param.data = init
    
    def _theta(self, x, cond=None, **kwargs): 
        return self._theta_param.unsqueeze(0).repeat(x.size(0), 1)

    def _invert(self, u, cond=None, log_abs_det=False, **kwargs): 
        theta = self._theta(u, cond=cond, **kwargs)
        
        return self.trnf(u, theta=theta, log_abs_det=log_abs_det, invert=True, **kwargs)

    
class ConditionerNet(Module):
    """Conditioner parameters network.

    Used to compute the parameters theta passed to a transformer.
    If its input_dim is 0 (i.e., a Flow of dimension 1 without conditioning),
    returns a learnable tensor containing the required result.
    """

    def __init__(
        self, 
        input_dim, output_dim, theta_init=None, net_f=None
    ):
        """
        Args:
            input_dim (int): input dimensionality.
            output_dim (int): output dimensionality. 
                Total dimension of all parameters combined.
            h_init (torch.Tensor): tensor of shape (output_dim,) to use as initializer.
                If None, original initialization used.
            net_f (function): function net_f(input_dim, output_dim, init=None)
                that creates a network with the given input and output dimensions,
                and possibly receives an init Tensor that,
                when not None, indicates that the function should be set 
                so that it always return that tensor. 
                This helps in initializing a tensor to the identity.
        """
        assert net_f is not None or not input_dim, 'Must provide a net_f if input_dim > 0'
        
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        if theta_init is not None:
            assert theta_init.shape == (output_dim,)
            
        if input_dim:
            self.net = net_f(input_dim, output_dim, init=theta_init)
        else:
            if theta_init is None:
                theta_init = torch.randn(output_dim)

            self.parameter = nn.Parameter(theta_init.unsqueeze(0))

    def forward(self, x):
        """Feed-forward pass."""
        if self.input_dim:
            return self.net(x)
        else:
            return self.parameter.repeat(x.size(0), 1)

    def warm_start(self, x):
        super().warm_start(x)
        
        if self.input_dim and hasattr(self.net, 'warm_start'):
            self.net.warm_start(x)

    
class AutoregressiveNaive(Conditioner):
    """Naive Autoregressive Conditioner.

    Implements a separate network for each dimension's theta parameters.
    """

    def __init__(self, trnf, net_f=None, **kwargs):
        """
        Args:
            trnf (flow.flow.Transformer): 
                transformer to use alongside this conditioner.
            net_f (class): function or class with input 
                (input_dim, output_dim, init=None) 
                that will be used by ConditionerNet 
                to create the Conditioner network.
        """
        super().__init__(trnf, **kwargs)

        theta_init = self.trnf._theta_init()

        self.nets = nn.ModuleList([
            ConditionerNet(
                idim + self.cond_dim, 
                self.trnf.theta_dims[idim], 
                theta_init=init, net_f=net_f
            )
            
            for idim, init in zip(
                range(self.dim),
                (
                    theta_init.view(self.dim, -1) 
                    if theta_init is not None 
                    else [None] * self.dim
                )
            )
        ])
    
    def _update_device(self, device):
        super()._update_device(device)
        
        for net in self.nets:
            net.device = device

    # Override methods
    def _theta(self, x, cond=None):
        x = prepend_cond(x, cond)

        return torch.cat([
            net(x[:, :self.cond_dim + i])
            for i, net in enumerate(self.nets)
        ], dim=1)

    def _invert(self, u, cond=None, log_abs_det=False, **kwargs):
        x = u
        
        # Invert dimension per dimension sequentially
        for i, net in enumerate(self.nets):
            last = i == len(self.nets) - 1
            
            if not last:
                x = x.detach().clone()
                
            theta = self._theta(x, cond=cond)
            x = self.trnf(u, theta, invert=True, log_abs_det=last and log_abs_det)
            
        # Note that x is (x, log_abs_det) if log_abs_det is True,
        # and x otherwise. So just return x no matter what
        return x

    def _warm_start(self, x, cond=None, **kwargs):
        xcond = prepend_cond(x, cond)
        for i, net in enumerate(self.nets):
            net.warm_start(xcond[:, :self.cond_dim + i])

        super()._warm_start(x, cond=cond, **kwargs)
        


# MADE
# ------------------------------------------------------------------------------

class MADE_Net(nn.Sequential):
    """Feed-forward network with masked linear layers used for MADE."""
            
    def __init__(
        self, 
        input_dim, output_dim, 
        hidden_sizes_mult=[10, 5], 
        act=nn.ReLU, use_batch_norm=True, dropout=0,
        cond_dim=0, theta_init=None,
    ):
        """
        Args:
            input_dim (int): number of inputs (flow dimension).
            output_dim (int): number of outputs (trnf.h_dim).
                Note that all parameters corresponding to a dimension
                are concatenated together. Therefore, for a 3D-affine trnf:
                    mu0, sigma0, mu1, sigma1, mu2, sigma2. 
            hidden_sizes_mult (list): multiplier w.r.t. input_size
                for each of the hidden layers.
            act (nn.Module): activation layer to use. 
                If None, no activation is done.
            cond_dim (int): dimensionality of the conditioning tensor, if any.
                non-negative int. If cond_dim > 0, the cond tensor is expected 
                to be concatenated before the input dimensions.
            theta_init (torch.Tensor): tensor to use as initializer 
                of the last layer bias. If None, original initialization used.
            """

        super().__init__()

        self.input_dim = input_dim
        self.theta_dim = self.output_dim = output_dim

        assert all(isinstance(m, int) and m >= 1 for m in hidden_sizes_mult)
        self.hidden_sizes = [ 
            (input_dim + (cond_dim > 0)) * m 
            for m in hidden_sizes_mult 
        ]

        assert isinstance(cond_dim, int) and cond_dim >= 0
        self.cond_dim = cond_dim

        # Add all layers to the Sequential module
        # Define mask connectivity
        m = [ 
            torch.arange(-cond_dim, input_dim) + 1
        ] + [
            (
                torch.arange(input_dim + (cond_dim > 0)) + 
                # torch.randperm(input_dim + (cond_dim > 0)) + 
                (cond_dim == 0)
            ).repeat(m) # 0 means cond input, > 0 are the actual inputs
            for m in hidden_sizes_mult
        ] + [ 
            torch.arange(1, input_dim + 1)\
                .view(-1, 1).repeat(1, output_dim // input_dim).flatten()
        ]

        # Create the actual layers
        for k, (m0, m1) in enumerate(zip(m[:-1], m[1:])):
            # Define masks
            h0, h1 = m0.size(0), m1.size(0)
            
            if k == 0 or use_batch_norm:
                self.add_module('batch_norm_%d' % k, nn.BatchNorm1d(h0))
            
            # Add a Dropout layer if dropout > .5
            if k > 0 and dropout > 0:
                self.add_module('dropout_%d' % k, nn.Dropout(dropout))

            # Check mask size
            if k == 0:
                assert h0 == input_dim + cond_dim
            elif k < len(self.hidden_sizes):
                assert (h0, h1) == tuple(self.hidden_sizes[k-1:k+1]), \
                    (h0, h1, self.hidden_sizes[k-1:k+1])
            else:
                assert h1 == output_dim

            # Create mask
            if k < len(self.hidden_sizes):
                mask = m1.unsqueeze(1) >= m0.unsqueeze(0)
            else:
                mask = m1.unsqueeze(1) > m0.unsqueeze(0)

            # Add it to the masked layer
            self.add_module(
                'masked_linear_%d' % k,
                MaskedLinear(h0, h1, bias=True, mask=mask)
            )

            # Add an activation if not the last layer
            if act is not None and k < len(m) - 2: # not the last one
                self.add_module(f'{act.__name__}_{k}', act())

        if theta_init is not None:
            for p in self.parameters():
                p.data = torch.randn_like(p) * 1e-3

            last_bias = self[-1].bias
            last_bias.data = theta_init.to(last_bias.device)

    def warm_start(self, x, **kwargs):
        bn = self[0]

        bn.running_mean = x.mean(0)
        bn.running_var = x.var(0)

        return self
    
    
class MADE_Resnet(MADE_Net):
    """Feed-forward network with masked linear layers used for MADE."""
            
    def __init__(self, *args, res_every=2, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert len(set(self.hidden_sizes[::res_every])) == 1
        self.res_every = res_every
        
    def forward(self, x):
        k = 0
        for n_layer, (name, layer) in enumerate(self.named_children()):
            x = layer(x)
            if name.startswith('masked_linear_'):
                k += 1
                if k == 1:
                    prev = x
                elif (k - 1) % self.res_every == 0 and n_layer < len(self) - 1:
                    x = x + prev
                    prev = x
                    
        return x


class MADE(Conditioner):
    """Masked Autoregressive flow for Density Estimation.

    http://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation
    """
        
    def __init__(self, trnf, net_f=MADE_Net, net_kwargs=None, **kwargs):
        """
        Args:
            trnf (flow.flow.Transformer): 
                transformer to use alongside this conditioner.
            net_f (nn.Module class): network to use for MADE. 
                Must use `MaskedLinear` layers with appropriate masks. 
                Defaults to `MADE_Net`.
        """

        super().__init__(trnf, **kwargs)

        net_kwargs = net_kwargs or {}

        self.net = net_f(
            self.dim, trnf.theta_dim, 
            cond_dim=self.cond_dim,
            theta_init=self.trnf._theta_init(),
            **net_kwargs
        )
        

    # Overrides:
    def _theta(self, x, cond=None):
        return self.net(prepend_cond(x, cond))

    def _invert(self, u, log_abs_det=False, cond=None, **kwargs):
        # Invert dimension per dimension sequentially
        x = torch.randn_like(u)

        for i in range(self.dim):
            self.net.train(self.training and i == self.dim - 1) # to avoid getting incorrect running means
            
            theta = self.net(prepend_cond(x, cond)) # obtain theta from previously computed x

            x = self.trnf(
                u, 
                theta, 
                invert=True, 
                log_abs_det=i == self.dim - 1 and log_abs_det,
                **kwargs
            )
        
        self.net.train(self.training) # reset self.net.training status
        
        return x # contains log_abs_det if log_abs_det was True

    def _warm_start(self, x, cond=None, **kwargs):
        self.net.warm_start(prepend_cond(x, cond), **kwargs)

        super().warm_start(x, cond=cond, **kwargs)
    
# ------------------------------------------------------------------------------


class _CouplingLayersTransformer(Flow):
    """Special Transformer used exclusively by CouplingLayers.

    Since Transformers and Conditioners need to be of the same size
    and CouplingLayers only passes a subset of dimensions to its Transformer,
    we need a Transformer that encapsulates the lower-dimensional Transformer.

    This will only transform the second split of dimensions, not the first one,
    but its dim is the whole dimensionality.
    """
    
    def __init__(self, trnf_cls, **kwargs):        
        super().__init__(**kwargs)
        
        self.trnf = trnf_cls(dim=self.dim // 2)
        self.theta_dims = (0,) * (self.dim - self.dim // 2) + self.trnf.theta_dims


    # Overrides
    def _activation(self, theta, **kwargs): 
        return self.trnf._activation(theta, **kwargs)

    def _transform(self, x, *theta, log_abs_det=False, **kwargs): 
        u = torch.zeros_like(x)
        d = self.trnf.dim
        
        t = self.trnf._transform(x[:, -d:], *theta, log_abs_det=log_abs_det, **kwargs)
        if log_abs_det:
            t, det = t
        u[:, :-d] = x[:, :-d]
        u[:, -self.trnf.dim:] = t
        
        if log_abs_det:
            return u, det
        else:
            return u

    def _invert(self, u, *theta, log_abs_det=False, **kwargs):
        x = torch.ones_like(u) * u
        d = self.trnf.dim
        
        t = self.trnf._invert(u[:, -d:], *theta, log_abs_det=log_abs_det, **kwargs)
        if log_abs_det:
            t, det = t
        x[:, :-d] = u[:, :-d]
        x[:, -d:] = t
        
        if log_abs_det:
            return x, det
        else:
            return x
        
    def _theta_init(self):
        # Return initialization values for pre-activation theta parameters.
        return self.trnf._theta_init()


class CouplingLayers(Conditioner):
    """CouplingLayers.

    https://arxiv.org/abs/1410.8516

    Simple implementation of CouplingLayers, where the tensor is divided 
    in two splits, one transformed with the identity, 
    the other with the given transformer. 
    The identity split has dim - dim // 2 dimensions,
    and the transformer one, dim // 2 dimensions.
    As such, the given trnf must have trnf.dim == cond.dim // 2.

    Pass both dimension keyword arguments (Transformer and Conditioner).
    Remember to apply a `flow.modules.Shuffle` flow before this Conditioner.
    """

    def __init__(self, trnf_cls, dim=1, net_f=None, **kwargs):
        """
        Args:
            trnf_cls (Flow class): transformer class to use on the second split.
                Pass a class, not an instanced transformer, as it will be instanced
                with half the dimensionality (only the second half needs it).
            dim (int): dimension of the Conditioner. 
            net_f (function): function(input_dim, output_dim, init=None).
        """
        assert dim > 1
        assert net_f is not None

        trnf = _CouplingLayersTransformer(trnf_cls, dim=dim)
        super().__init__(trnf, dim=dim, **kwargs)

        self.theta_net = net_f(
            dim - self.dim // 2 + self.cond_dim, trnf.theta_dim, 
            init=self.trnf._theta_init()
        )

    # Overrides
    def _theta(self, x, cond=None, **kwargs): 
        return self.theta_net(prepend_cond(x[:, :-(self.dim // 2)], cond))

    def _invert(self, u, cond=None, log_abs_det=False, **kwargs): 
        theta = self.theta_net(prepend_cond(u[:, :-(self.dim // 2)], cond))
        
        return self.trnf(u, theta, invert=True, log_abs_det=log_abs_det, **kwargs)

    def _warm_start(self, x, cond=None, **kwargs):
        if hasattr(self.theta_net, 'warm_start'):
            self.theta_net.warm_start(prepend_cond(x[:, :-(self.dim // 2)], cond), **kwargs)
        
        super()._warm_start(x, cond=cond, **kwargs)
        
        
class DAG_Conditioner(Conditioner):
    
    def __init__(self, *args, A=None, net_f=None, **kwargs):
        """
        Args:
            A (torch.Tensor): 
                upper-diagonal (with 0s at diagonal) 
                boolean matrix with shape (self.dim, self.dim)
                where A[i, j] == True means that i->j in the graph.
                If None, autoregressive matrix is assumed (A[i, j] == (i < j)).
            net_f (function):
                function(input_dim, output_dim, init=None) that returns the requested network.
        """
        super().__init__(*args, **kwargs)
        
        if A is None:
            # Autoregressive
            A = torch.zeros(self.dim, self.dim, dtype=bool)
            i1, i2 = torch.triu_indices(*A.shape, offset=1)
            A[i1, i2] = True
        else:
            i1, i2 = torch.tril_indices(*A.shape)
            assert A is not None and \
                A.shape == (self.dim, self.dim) and \
                A.dtype == torch.bool and \
                not A[i1, i2].any().item()
        
        nodes = list(range(A.size(1)))
        
        parents = { i: set() for i in range(self.dim) }
        for j in range(self.dim):
            for i in range(j):
                if A[i, j].item():
                    parents[j].add(i)
        
        self.order = topological_order(nodes, parents, return_levels=True)
        self.register_buffer('A', A)
        
        # Repeat A columns as many times as required for each parameter
        # TODO: Correct this when h_dim_1d becomes a list        
        A = torch.cat([
            A[:, [i]].repeat(1, m)
            for i, m in enumerate(self.trnf.theta_dims)
        ], 1)
        
        if self.cond_dim:
            A = torch.cat([ torch.ones(self.cond_dim, A.size(1), dtype=bool), A ], 0)
        
        self.net = AdjacencyMaskedNet(A, net_f=net_f, init=self.trnf._theta_init())        
        
    def _theta(self, x, cond=None, **kwargs): 
        # Return the (non-activated) tensor of parameters h 
        # corresponding to the given x. If this is a conditional flow,
        # the conditioning tensor is passed as the 'cond' kwarg.
        
        return self.net(prepend_cond(x, cond))

    def _invert(self, u, cond=None, log_abs_det=False, **kwargs): 
        # Transform u into x.
        x = torch.zeros_like(u)
        
        for _ in range(len(self.order) - 1):
            with torch.no_grad():
                h = self._theta(x, cond=cond, **kwargs)
                x = self.trnf(u, h, invert=True, log_abs_det=False, **kwargs)
        
        # Finally, compute normally
        h = self._theta(x, cond=cond, **kwargs)
        return self.trnf(u, h, invert=True, log_abs_det=log_abs_det, **kwargs)
    
    def _warm_start(self, *args, **kwargs):
        if hasattr(self.net, 'warm_start'):
            self.net.warm_start(*args, **kwargs)
        
        super()._warm_start(*args, **kwargs)