"""
Tests all transformers in flow.transformer.
"""

import pytest

import numpy as np
import torch

from flow.transformer import Affine, NonAffine, DSF, Q_Spline, RQ_Spline
from flow import prior

from utils import torch_eq_float, skip_cuda


# Fill this list with all conditioners you want to test.
test_transformers = [
    Affine,
    NonAffine,
    DSF,
    Q_Spline,
    RQ_Spline,
]

# For some transformers, a specific prior is required.
# If it isn't in the dict, it's assumed to be prior.Normal
trnf_prior = {
    Q_Spline: prior.Uniform
}


@pytest.mark.parametrize('trnf', test_transformers)
@torch.no_grad()
def test_shape_and_return_requirements(trnf, dim=2):
    trnf = trnf(dim=dim, prior=trnf_prior.get(trnf, prior.Normal))

    # Generate 10 samples in the appropiate domain for X
    u = trnf.prior.sample(10) # get them from the transformer's prior
    
    # Generate any random pre-activation parameters
    _h = torch.randn(u.size(0), trnf.h_dim)
    h = trnf._activation(_h)

    # And transform our prior sample to X
    x = trnf(u, _h, invert=True)

    # We'll use this sample from now on

    assert isinstance(h, (tuple, list)), \
        'Parameters returned by _activation need to be set in a tuple'
    assert all([hi.shape[0] == x.size(0) for hi in h]), \
        'Parameters returned by _activation need to be of the same size as x'

    # We'll call trnf with and without log_det 
    # to make sure both options are considered.
    u = trnf(x, _h)
    _, log_det = trnf(x, _h, log_det=True)

    assert u.shape == x.shape
    assert log_det.shape == (x.size(0),)

    x2 = trnf(u, _h, invert=True)
    _, log_det = trnf(u, _h, invert=True, log_det=True)

    assert x2.shape == x.shape
    assert log_det.shape == (x.size(0),)

    h_init = trnf._h_init()
    assert h_init is None or (
        isinstance(h_init, torch.Tensor) and
        h_init.shape == (trnf.h_dim,) and
        h_init.device == trnf.device
    )