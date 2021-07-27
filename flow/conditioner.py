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

from .flow import Conditioner
from .utils import Module, MultiHeadNet, prepend_cond


def _basic_net(input_dim, output_dim, hidden_dim=(100, 50), nl=nn.ReLU, init=None):
    assert isinstance(hidden_dim, (tuple, list)) 
    assert all(isinstance(x, int) and x > 0 for x in hidden_dim)

    n_layers = len(hidden_dim) + 1

    def create_layer(n_layer, n_step):
        if n_step == 0:
            i = input_dim if n_layer == 0 else hidden_dim[n_layer - 1]
            o = output_dim if n_layer == n_layers - 1 else \
                hidden_dim[n_layer]

            return nn.Linear(i, o)
        else:
            if n_layer == n_layers - 1:
                return None
            else:
                return nl()

    net = nn.Sequential(
        nn.BatchNorm1d(input_dim, affine=False),
        *(
            layer
            for layer in (
                create_layer(n_layer, n_step)
                for n_layer in range(n_layers)
                for n_step in range(2)
            )
            if layer is not None
        )
    )

    if init is not None:
        # Initialize all weights and biases to close to 0,
        # and the last biases to init
        for p in net.parameters():
            p.data = torch.randn_like(p) * 1e-3

        last_bias = net[-1].bias
        last_bias.data = init.to(last_bias.device)
        
    return net


class ConditionerNet(Module):
    """Conditioner parameters network.

    Used to compute the parameters h passed to a transformer.
    If called with a void tensor (in an autoregressive setting, the first step),
    returns a learnable tensor containing the required result.
    """

    def __init__(
        self, 
        input_dim, output_dim, net_f, h_init=None,
    ):
        """
        Args:
            input_dim (int): input dimensionality.
            output_dim (int): output dimensionality. 
                Total dimension of all parameters combined.
            net_f (function): function net_f(input_dim, output_dim, init=None)
                that creates a network with the given input and output dimensions,
                and possibly receives an init Tensor that,
                when not None, indicates that the function should be set 
                so that it always return that tensor. 
                This helps in initializing a tensor to the identity.
            h_init (torch.Tensor): tensor to use as initializer 
                of the last layer bias. If None, original initialization used.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if h_init is not None:
            assert h_init.shape == (output_dim,)
            
        if input_dim:
            self.net = net_f(input_dim, output_dim, init=h_init)
        else:
            if h_init is None:
                h_init = torch.randn(output_dim)

            self.parameter = nn.Parameter(h_init.unsqueeze(0))

    def forward(self, x):
        """Feed-forward pass."""
        if self.input_dim:
            return self.net(x)
        else:
            return self.parameter.repeat(x.size(0), 1) # n == batch size

    def warm_start(self, x, **kwargs):           
        if self.input_dim:
            bn = self.net[0]
            assert bn.__class__.__name__.startswith('BatchNorm')

            bn.running_mean.data = x.mean(0)
            bn.running_var.data = x.var(0)

        return self


class AutoregressiveNaive(Conditioner):
    """Naive Autoregressive Conditioner.

    Implements a separate network for each dimension's h parameters.
    """

    def __init__(self, trnf, net_f, **kwargs):
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

        h_init = self.trnf._h_init()

        self.nets = nn.ModuleList([
            ConditionerNet(idim + self.cond_dim, self.trnf.h_dim, h_init=init, net_f=net_f)
            for idim, init in zip(
                range(self.dim),
                (
                    h_init.view(self.dim, -1) 
                    if h_init is not None 
                    else [None] * self.dim
                )
            )
        ])
    
    def _update_device(self, device):
        super()._update_device(device)
        
        for net in self.nets:
            net.device = device

    # Override methods
    def _h(self, x, cond=None, start=0):
        assert 0 <= start and start <= min(self.dim - 1, x.size(1))
        x = prepend_cond(x, cond)

        return torch.cat([
            net(x[:, :self.cond_dim + i])
            for i, net in enumerate(self.nets[start:], start)
            if (x.size(1) == self.cond_dim and i == 0) or 
                i <= x.size(1) - self.cond_dim
        ], dim=1)

    def _invert(self, u, cond=None, log_det=False, **kwargs):
        x = u[:, :0] # (n, 0) tensor
        
        # Invert dimension per dimension sequentially
        log_det_i = torch.zeros_like(u[:, 0])
        for i, net in enumerate(self.nets):
            # Obtain h_i from previously computed x
            h_i = self._h(x, cond=cond, start=i)
            x_i = self.trnf(u[:, [i]], h_i, invert=True, log_det=log_det)
            if log_det:
                x_i, log_det_i2 = x_i
                log_det_i = log_det_i + log_det_i2

            x = torch.cat([x, x_i], 1)

        if log_det:
            return x, log_det_i
        else:
            return x

    def warm_start(self, x, cond=None, **kwargs):
        super().warm_start(x, cond=cond, **kwargs)

        x = prepend_cond(x, cond)
        for i, net in enumerate(self.nets):
            net.warm_start(x[:, :self.cond_dim + i])

        return self


# MADE
# ------------------------------------------------------------------------------


class MaskedLinear(nn.Linear):
    """Extend `torch.nn.Linear` to use a boolean mask on its weights."""

    def __init__(self, in_features, out_features, bias=True, mask=None):
        """
        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sampl.
            bias (bool): If set to False, 
                the layer will not learn an additive bias. 
            mask (torch.Tensor): boolean mask to apply to the weights.
                Tensor of shape (out_features, in_features).
                If None, defaults to an unmasked Linear layer.
        """
        super().__init__(in_features, out_features, bias)
        
        if mask is None:
            mask = torch.ones(out_features, in_features, dtype=bool)
        else:
            mask = mask.bool()

        assert mask.shape == (out_features, in_features)
        self.register_buffer('mask', mask)

    def forward(self, input):
        """Extend forward to include the buffered mask."""
        return F.linear(input, self.mask * self.weight, self.bias)

    def set_mask(self, mask):
        """Set the buffered mask to the given tensor."""
        self.mask.data = mask


class MADE_Net(nn.Sequential):
    """Feed-forward network with masked linear layers used for MADE."""
            
    def __init__(
        self, 
        input_dim, output_dim, 
        hidden_sizes_mult=[10, 5], 
        act=nn.ReLU, use_batch_norm=True, dropout=0,
        cond_dim=0, h_init=None,
    ):
        """
        Args:
            input_dim (int): number of inputs (flow dimension).
            output_dim (int): number of outputs (trnf.dim * trnf.h_dim).
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
            h_init (torch.Tensor): tensor to use as initializer 
                of the last layer bias. If None, original initialization used.
            """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        assert not output_dim % input_dim
        self.h_dim = output_dim // input_dim

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
                .view(-1, 1).repeat(1, self.h_dim).flatten()
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

        if h_init is not None:
            for p in self.parameters():
                p.data = torch.randn_like(p) * 1e-3

            last_bias = self[-1].bias
            last_bias.data = h_init.to(last_bias.device)

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
        
    def __init__(self, trnf, net=MADE_Net, net_kwargs=None, **kwargs):
        """
        Args:
            trnf (flow.flow.Transformer): 
                transformer to use alongside this conditioner.
            net (nn.Module class): network to use for MADE. 
                Must use `MaskedLinear` layers with appropriate masks. 
                Defaults to `MADE_Net`.
        """

        super().__init__(trnf, **kwargs)

        net_kwargs = net_kwargs or {}

        self.net = net(
            self.dim, self.dim * trnf.h_dim, 
            cond_dim=self.cond_dim,
            h_init=self.trnf._h_init(),
            **net_kwargs
        )
        

    # Overrides:
    def _h(self, x, cond=None):
        return self.net(prepend_cond(x, cond))

    def _invert(self, u, log_det=False, cond=None, **kwargs):
        # Invert dimension per dimension sequentially
        x = prepend_cond(torch.randn_like(u), cond)
        log_det_sum = torch.zeros_like(u[:, 0])

        for i in range(self.dim):
            self.net.train(self.training and i == self.dim) # to avoid getting incorrect running means
            h_i = self.net(x) # obtain h_i from previously computed x

            res = self.trnf(
                u[:, [i]], 
                h_i[:, self.trnf.h_dim * i : self.trnf.h_dim * (i + 1)], 
                invert=True, 
                log_det=log_det,
                **kwargs
            )

            if log_det:
                xi, log_det_i = res
                log_det_sum = log_det_sum + log_det_i
            else:
                xi = res

            x[:, [self.cond_dim + i]] = xi
        
        self.net.train(self.training) # reset self.net.training status
        
        # Remove the cond part of x
        x = x[:, self.cond_dim:]        
            
        if log_det:
            return x, log_det_sum
        else:
            return x

    def warm_start(self, x, cond=None, **kwargs):
        super().warm_start(x, cond=cond, **kwargs)
        self.net.warm_start(prepend_cond(x, cond), **kwargs)

        return self
# ------------------------------------------------------------------------------


from flow.flow import Transformer
from flow.modules import Shuffle

class _CouplingLayersTransformer(Transformer):
    """Special Transformer used exclusively by CouplingLayers.

    Since Transformers and Conditioners need to be of the same size
    and CouplingLayers only passes a subset of dimensions to its Transformer,
    we need a Transformer that encapsulates the lower-dimensional Transformer.

    This will only transform the second split of dimensions, not the first one,
    but its dim is the whole dimensionality.
    """
    
    def __init__(self, trnf, dim=None):
        assert dim is not None and dim in (trnf.dim * 2, trnf.dim * 2 + 1)
        
        super().__init__(dim=dim, h_dim=trnf.h_dim)

        self.trnf = trnf


    # Overrides
    def _activation(self, h, **kwargs): 
        return self.trnf._activation(h, **kwargs)

    def _transform(self, x, *h, log_det=False, **kwargs): 
        # CouplingLayers will pass the whole Tensor to _transform.
        assert x.size(1) == self.dim

        x1, x2 = x[:, :self.trnf.dim], x[:, self.trnf.dim:]
        u1 = x1
        res2 = self.trnf._transform(x2, *h, log_det=log_det, **kwargs)

        if log_det:
            u2, log_det = res2

            return torch.cat([u1, u2], 1), log_det
        else:
            u2 = res2

            return torch.cat([u1, u2], 1)

    def _invert(self, u2, *h, log_det=False, **kwargs):
        # CouplingLayers will only pass the transformer split to _invert.
        assert u2.size(1) == self.trnf.dim

        res2 = self.trnf._invert(u2, *h, log_det=log_det, **kwargs)

        if log_det:
            x2, log_det = res2

            return x2, log_det
        else:
            x2 = res2

            return x2

    def _h_init(self):
        # Return initialization values for pre-activation h parameters.
        return self.trnf._h_init()


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

    def __init__(self, trnf, dim=None, net_kwargs=None, **kwargs):
        """
        Args:
            trnf (Transformer): transformer to use on the second split.
            dim (int): dimension of the Conditioner. 
                Note that its transformer must have dim // 2 dimensionality.
        """

        assert dim is not None and dim >= 2, 'Must pass dim to CouplingLayers'
        assert trnf.dim == dim // 2

        trnf = _CouplingLayersTransformer(trnf, dim=dim)
        super().__init__(trnf, dim=dim, **kwargs)

        self.h_net = ConditionerNet(
            dim - dim // 2 + self.cond_dim, dim // 2 * trnf.h_dim, 
            h_init=self.trnf._h_init(),
            **(net_kwargs or {})
        )


    # Overrides
    def _h(self, x, cond=None, **kwargs): 
        id_dim = self.dim - self.dim // 2
        x1 = x[:, :id_dim]
        
        return self.h_net(prepend_cond(x1, cond))

    def _invert(self, u, cond=None, log_det=False, **kwargs): 
        id_dim = self.dim - self.dim // 2
        u1, u2 = u[:, :id_dim], u[:, id_dim:]

        x1 = u1        
        h = self.h_net(prepend_cond(x1, cond))
        res2 = self.trnf(u2, h, invert=True, log_det=log_det, **kwargs)

        if log_det:
            x2, log_det = res2

            return torch.cat([x1, x2], 1), log_det
        else:
            x2 = res2

            return torch.cat([x1, x2], 1)

    def warm_start(self, x, cond=None, **kwargs):
        super().warm_start(x, cond=cond, **kwargs)

        id_dim = self.dim - self.dim // 2
        x1 = x[:, :id_dim]
        x1 = prepend_cond(x1, cond)
        self.h_net.warm_start(x1, **kwargs)

        return self