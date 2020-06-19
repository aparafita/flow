"""
Tests all priors in flow.prior.
"""

import pytest

import numpy as np
import torch

from flow.prior import Normal

from utils import skip_cuda, no_grad_dec


# Fill this list with all priors you want to test.
test_priors = [
    Normal
]


@pytest.mark.parametrize('prior', test_priors)
@no_grad_dec
def test_prior(prior, dim=2):
    prior = prior(dim=dim)

    sample = prior.sample(10)
    assert sample.shape == (10, dim)

    nll = prior.nll(sample)
    assert nll.shape == (10,)


@pytest.mark.parametrize('prior', test_priors)
@skip_cuda
@no_grad_dec
def test_prior_device(prior, dim=2):
    prior = prior(dim=dim)

    prior.cuda()
    sample = prior.sample(10)
    assert sample.device == prior.device and prior.device.type == 'cuda'

    prior.cpu()
    sample = prior.sample(10)
    assert sample.device == prior.device and prior.device.type == 'cpu'