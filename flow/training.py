"""
Train utilities for flows.

Includes functions:

* `get_device`: get the default torch.device (cuda if available).
* `train`: used to train flows with early stopping.
* `plot_losses`: plot training and validation losses from a `train` session.
* `test_nll`: compute the test negative-loglikelihood of the test set.
"""


from tempfile import TemporaryFile
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
from torch import nn, optim

from . import Flow

from torch_utils.training import *


def loss_f(flow, batch):
    """Training loss based on flow.nll.
    
    Assumes that batch is a tuple (tensor,) or (tensor, cond).
    Sends tensor to flow.device and calls nll(tensor, cond=cond).
    """
    if len(batch) == 1:
        batch, = batch
        batch = batch.to(flow.device)
        cond = None
    else:
        assert len(batch) == 2
        batch, cond = batch
        batch = batch.to(flow.device)
        cond = cond.to(flow.device)
    
    return flow.nll(batch, cond=cond)