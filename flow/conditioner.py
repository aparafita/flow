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
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError('cond can\'t be None if cond_dim > 0')
            if cond.size(1) != self.cond_dim:
                raise ValueError(
                    f'cond dim ({cond.size(1)}) '
                    'doesn\'t fit cond_dim ({self.cond_dim})'
                )

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