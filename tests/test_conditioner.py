"""
Tests all conditioners in flow.conditioner.
"""

import pytest

import torch

from flow.conditioner import AutoregressiveNaive, MADE
from flow.transformer import Affine

from utils import torch_eq_float, skip_cuda


# Fill this list with all conditioners you want to test.
test_conditioners = [
    AutoregressiveNaive,
    MADE,
]


@skip_cuda
@pytest.mark.parametrize('cond', test_conditioners)
def test_device(cond, dim=2):
    """Test that device is updated across a Conditioner flow."""
    flow = cond(Affine(dim=dim), dim=dim)

    for method in [flow.cuda, flow.cpu]:
        method() # flow.cuda() or flow.cpu()
        assert flow.trnf.device == flow.device
        assert all(p.device == flow.device for p in flow.parameters())


@pytest.mark.parametrize('cond', test_conditioners)
@torch.no_grad()
def test_h_shape(cond, dim=2):
    """Test that _h returns the correct parameters shape."""
    flow = cond(Affine(dim=dim), dim=dim)

    x = torch.randn(10, flow.dim)
    h = flow._h(x)

    assert h.shape == (x.size(0), flow.trnf.h_dim * flow.dim)


@pytest.mark.parametrize('cond', test_conditioners)
@torch.no_grad()
def test_conditional(cond, dim=2, cond_dim=3, seed=123):
    """Test that conditional Conditioner works correctly."""
    assert cond_dim > 0, 'Don\'t call this test with cond_dim <= 0'

    trnf = Affine(dim=dim)

    if not cond.conditional:
        with pytest.raises(ValueError):
            flow = cond(trnf, cond_dim=cond_dim)

        return # nothing else to do here

    flow = cond(trnf, cond_dim=cond_dim)

    # Check that calling it without cond raises a ValueError
    x = torch.zeros(2, flow.dim)
    with pytest.raises(ValueError):
        flow(x, cond=None)

    with pytest.raises(ValueError):
        flow(x, cond=None, invert=True)

    # Check that calling it with cond of wrong dimension also raises ValueError
    cond = torch.zeros(x.size(0), cond_dim + 1)
    with pytest.raises(ValueError):
        flow(x, cond=cond)

    with pytest.raises(ValueError):
        flow(x, cond=cond, invert=True)
    
    # TODO: Check that calling it with different conds affects the output.
    # Note that depending on the flow, its initialization 
    # and the values sampled for cond, their result might be equal.
    torch.random.manual_seed(seed)
    cond = torch.randn(x.size(0), cond_dim)
    cond2 = torch.randn_like(cond)

    u = flow(x, cond=cond)
    u2 = flow(x, cond=cond2)

    # assert not torch_eq_float(u, u2, eps=1e-6)


def test_made(dim=2, cond_dim=3):
    flow = MADE(Affine(dim=dim), cond_dim=cond_dim)

    x = torch.randn(1, dim).requires_grad_(True)
    cond = torch.randn(1, cond_dim).requires_grad_(True)


    # In all cases, cond should receive a gradient.
    # The gradient coming from the 1st dimension's parameters should be 0
    h = flow._h(x, cond=cond)
    h0 = h[:, :flow.trnf.h_dim]

    h0.sum(1).backward()

    assert x.grad[0, 0] == 0
    assert x.grad[0, 1] == 0
    assert (cond.grad != 0).all()


    # The gradient coming from the 2nd dimension's parameters should be 0
    # only for the 2nd dimension, not the first one.
    x.grad, cond.grad = None, None
    h = flow._h(x, cond=cond)
    h1 = h[:, flow.trnf.h_dim:]

    h1.sum(1).backward()

    assert x.grad[0, 0] != 0
    assert x.grad[0, 1] == 0
    assert (cond.grad != 0).all()