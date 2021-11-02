"""
Train utilities for flows. Based on `torch_misc.training`. 

Includes a predefined loss_function: `loss_f` 
and all necessary functions from `torch_misc.training`.
"""

from torch_misc.training import *


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