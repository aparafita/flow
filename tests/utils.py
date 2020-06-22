"""
Contains utilities for testing.
"""

import pytest

import torch


def torch_eq_float(x, y, eps=1e-3):
    """Compare all entries in x and y for equality, up to eps differences."""
    return ((x - y).abs() < eps).all().item()

def skip_cuda(test):
    """Decorator to skip a test if it requires cuda."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(), 
        reason='We need cuda for this test'
    )(test)