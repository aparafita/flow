"""
Implementations for Flow-Conditioners.
"""

import numpy as np

import torch
from torch import nn

import torch.nn.functional as F

from .flow import Conditioner


def default_net(
    input_dim, output_dim, n_layers=3,
    hidden_dim=(100, 50), activation=nn.ReLU, 
):
    """Create a basic feed-forward network.

    Args:
        input_dim (int): input dimensionality.
        output_dim (int): output dimensionality.
        n_layers (int): number of layers in the network,
            counting input and output layers.
            For example, input + hidden + output is n_layers = 3.
        hidden_dim (list): list with the dimensionality of each hidden layer.
            Must contain n_layers - 1 non-negative ints.
        activation (torch.nn.Module): Module class to instantiate 
            as activation layer after each linear layer.
    """

    assert isinstance(hidden_dim, (tuple, list)) 
    assert len(hidden_dim) == n_layers - 1
    assert all(isinstance(x, int) and x > 0 for x in hidden_dim)

    def create_layer(n_layer, n_step):
        if n_step == 0:
            i = input_dim if n_layer == 0 else hidden_dim[n_layer - 1]
            o = output_dim if n_layer == n_layers - 1 else hidden_dim[n_layer]

            return nn.Linear(i, o)
        else:
            if n_layer == n_layers:
                return None
            else:
                return activation()

    return nn.Sequential(*(
        layer
        for layer in (
            create_layer(n_layer, n_step)
            for n_layer in range(n_layers)
            for n_step in range(2)
        )
        if layer is not None
    ))


class ConditionerNet(nn.Module):
    """Conditioner parameters network.

    Used to compute the parameters h passed to a transformer.
    If called with a void tensor (in an autoregressive setting, the first step),
    returns a learnable tensor containing the required result.
    """

    def __init__(
        self, 
        input_dim, output_dim, net=default_net, params_init=torch.randn
    ):
        """
        Args:
            input_dim (int): input dimensionality.
            output_dim (int): output dimensionality. 
                Total dimension of all parameters combined.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if input_dim:
            self.net = net(input_dim, output_dim)
        else:
            self.parameter = nn.Parameter(params_init(1, output_dim))

    def forward(self, x):
        """Feed-forward pass."""
        if self.input_dim:
            return self.net(x)
        else:
            return self.parameter.repeat(x.size(0), 1) # n == batch size


class AutoregressiveNaive(Conditioner):
    """Naive Autoregressive Conditioner.

    Implements a separate network for each dimension's h parameters.
    """

    def __init__(self, trnf, net=None, **kwargs):
        """
        Args:
            net (class): torch.nn.Module class that computes the parameters.
                If None, defaults to `ConditionerNet` with `default_net`.
        """
        super().__init__(trnf, **kwargs)

        if net is None:
            net = ConditionerNet

        self.nets = nn.ModuleList([
            net(idim + self.cond_dim, self.trnf.h_dim)
            for idim in range(self.dim)
        ])

    # Override methods
    def _h(self, x, cond=None, start=0):
        assert 0 <= start and start <= min(self.dim - 1, x.size(1))
        x = self._prepend_cond(x, cond)

        return torch.cat([
            net(x[:, :self.cond_dim + i])
            for i, net in enumerate(self.nets[start:], start)
            if (x.size(1) == self.cond_dim and i == 0) or 
                i <= x.size(1) - self.cond_dim
        ], dim=1)

    def _invert(self, u, cond=None, log_det=False):
        x = u[:, :0] # (n, 0) tensor
        
        # Invert dimension per dimension sequentially
        for i, net in enumerate(self.nets):
            # Obtain h_i from previously computed x
            h_i = self._h(x, cond=cond, start=i)
            x_i = self.trnf(u[:, [i]], h_i, invert=True)
            x = torch.cat([x, x_i], 1)

        if log_det:
            h = self._h(x, cond=cond)
            _, log_det = self.trnf(u, h, log_det=True, invert=True)
            return x, log_det
        else:
            return x


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
        return F.linear(input, self.mask * self.weight, self.bias)

    def set_mask(self, mask):
        self.mask.data = mask


class MADE_Net(nn.Sequential):
    """Feed-forward network with masked linear layers used for MADE."""
            
    def __init__(
        self, 
        input_dim, output_dim, 
        hidden_sizes_mult=[10, 5], act=nn.ReLU,
        cond_dim=0
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
                torch.randperm(input_dim + (cond_dim > 0)) + 
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


class MADE(Conditioner):
    """Masked Autoregressive flow for Density Estimation.

    http://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation
    """
        
    def __init__(self, trnf, net=MADE_Net, **kwargs):
        """
        Args:
            dim (int): flow dimension.
            h_dim (int): number of parameters per dimension.
            net (nn.Module class): network to use for MADE. 
                Must use `MaskedLinear` layers. Defaults to `MADE_Net`.
        """

        super().__init__(trnf, **kwargs)

        self.net = net(self.dim, self.dim * trnf.h_dim, cond_dim=self.cond_dim)
        

    # Overrides:
    def _h(self, x, cond=None, **kwargs):
        return self.net(self._prepend_cond(x, cond))

    def _invert(self, u, log_det=False, cond=None, **kwargs):
        # Invert dimension per dimension sequentially
        # We don't use gradients now; we'll do a gradient run right after
        with torch.no_grad():
            x = self._prepend_cond(torch.randn_like(u), cond)

            for i in range(self.dim):
                h_i = self.net(x) # obtain h_i from previously computed x
                
                x[:, [self.cond_dim + i]] = trnf(
                    u[:, [i]], 
                    h_i[:, self.h_dim * i : self.h_dim * (i + 1)], 
                    invert=True, 
                    log_det=False
                )

        # Run again, this time to get the gradient and log_det if required
        h = self.net(x) # now we can compute for all dimensions
        
        return trnf(u, h, invert=True, log_det=log_det)

# ------------------------------------------------------------------------------