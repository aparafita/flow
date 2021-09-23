"""
Tests basic abstract classes in flow to work as intended.
Checks inheritance mechanisms and basic funcionality in each class.
"""

import pytest

import numpy as np
import torch

from torch import nn

from flow.flow import Flow, inv_flow, Sequential
from flow import prior
from flow.modules import Affine as AffineFlow

from utils import torch_eq_float, skip_cuda


@torch.no_grad()
def test_flow_affine():
    """Test if Flow inheritance requirements are preserved.

    We'll run all Flow functions with the modules.AffineFlow flow.
    If flow.Flow has changed, AffineFlow should stop working.
    """ 
    weight = torch.ones(1, 1) * 2
    flow = AffineFlow(weight=weight, prior=prior.Normal)
    x = torch.ones(1, 1)

    u, log_det = flow(x, log_det=True)
    assert log_det.shape == (1,)
    assert torch_eq_float(u, 2)
    assert torch_eq_float(log_det, np.log(2))

    x2, log_det = flow(u, invert=True, log_det=True)
    assert log_det.shape == (1,)
    assert torch_eq_float(x2, 1.)
    assert torch_eq_float(log_det, -np.log(2))

    assert flow.sample(1).shape == (1, 1) # pass sample
    expected_nll = ( # -log_f(u) - log_det
        .5 * (np.log(2 * np.pi) + (u ** 2).sum(dim=1)) - # -log_f(u)
        np.log(2) # log_det
    )
    assert torch_eq_float(flow.nll(x), expected_nll) # pass nll


@torch.no_grad()
def test_inv_flow():
    """Test if inv_flow inverts a Flow class."""
    weight = torch.ones(1, 1) * 2
    flow = AffineFlow(weight=weight, prior=prior.Normal)
    flow_inv = inv_flow(AffineFlow)(weight=weight, prior=prior.Normal)

    x = flow.sample(100)
    u = flow(x)
    
    x2 = flow_inv(u)
    u2 = flow_inv(x, invert=True)

    assert torch_eq_float(x, x2)
    assert torch_eq_float(u, u2)


@torch.no_grad()
def test_sequential():
    """Test all operations of a Sequential"""
    flows = [
        AffineFlow(weight=torch.ones(1, 1) * 2.)
        for _ in range(5)
    ]

    # Add an index attribute to each flow to distinguish them for the next test
    for i, flow in enumerate(flows):
        flow.index = i

    flow = Sequential(*flows, prior=prior.Normal) # transform: x * 2^5

    # Test __getitem__
    for i in range(5):
        assert flow[i].index == i

    for i in range(-5, 0):
        assert flow[i].index == 5 + i

    for i in range(5):
        for j in range(1, 6):
            for k in range(j - i):
                assert flow[i:j][k].index == i + k

    for i in range(0, 2):
        assert flow[::2][i].index == 2 * i

    # Test __iter__
    for i, f in enumerate(flow):
        assert f.index == flow[i].index

    # Test _transform and _invert
    weight = 2 ** 5

    x = torch.ones(1, 1)
    u, log_det = flow(x, log_det=True) # should be 2 ** 5
    x2, log_det_inv = flow(x, invert=True, log_det=True) # should be 2 ** -5

    assert torch_eq_float(u, torch.ones_like(u) * weight)
    assert torch_eq_float(x2, torch.ones_like(u) / weight)
    assert torch_eq_float(log_det, 5 * np.log(2))
    assert torch_eq_float(log_det, -log_det_inv)

    # Sequential._update_device is tested in test_sequential_device


@skip_cuda
def test_sequential_device():
    """Test that device is updated across a whole flow."""
    
    flow = Sequential(*(AffineFlow() for _ in range(5)), prior=prior.Normal)

    for method in [flow.cuda, flow.cpu]:
        method() # flow.cuda() or flow.cpu()
        for f in flow:
            assert f.device == flow.device

        assert all(p.device == flow.device for p in flow.parameters())