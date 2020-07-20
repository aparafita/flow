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

from tqdm import tqdm

import torch
from torch import nn, optim

from . import Flow


def get_device():
    """Return default cuda device if available, cpu otherwise."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(
    flow, trainX, valX, cond_train=None, cond_val=None, loss_f=None,
    post_training_f=None, post_validation_f=None,
    batch_size=32, optimizer=optim.Adam, 
    optimizer_kwargs=dict(lr=1e-3, weight_decay=1e-3),
    n_epochs=int(1e6), patience=100, gradient_clipping=None
):
    r"""Train Flow model with (optional) early stopping.

    Can KeyboardInterrupt safely; 
    the resulting model will be the best one before the interruption.

    Args:
        flow (Flow): flow to train.
        
        trainX (torch.Tensor): training dataset.
        valX (torch.Tensor): validation dataset.

        cond_train (torch.Tensor): conditioning tensor for trainX.
            If None, non-conditional flow assumed.
        cond_val (torch.Tensor): conditioning tensor for valX.
            If None, non-conditional flow assumed.

        loss_f (func): function(batch, idx, cond=None) to use as loss. 
            If None, uses flow.nll(batch, cond=cond) instead.

            idx is an index tensor signaling which entries in trainX or valX
            (depending on whether flow.training is True) are contained in batch.
            cond is an optional keyword argument with the conditioning tensor,
            if the flow is conditional. Otherwise, it's just None 
            and should be ignored.
            Returns a tensor with the loss computed for each entry in the batch.
            
        
        batch_size (int or float): If float, ratio of trainX to use per batch.
            If int, batch size.
        optimizer (torch.optim.Optimizer): optimizer class to use.
        optimizer_kwargs (dict): kwargs to pass to the optimizer.

        n_epochs (int): maximum number of epochs for training.
        patience (int): maximum number of epochs with no improvement
            in validation loss before stopping. 
            To avoid using early stopping, set to 0.

    Returns:
        train_losses: list with entries (float(epoch), loss).
        val_losses: list with entries (epoch, loss).

    The results of this function can be passed to `plot_losses` directly.
    """
    
    assert isinstance(flow, Flow)
    assert flow.prior is not None, 'flow.prior is required'

    conditional = cond_train is not None or cond_val is not None
    if conditional:
        assert (cond_train is not None and cond_val is not None), \
            'If flow is conditional, pass cond_train and cond_val'
    else:
        cond = None # let's just leave it as a None for later

    if isinstance(batch_size, float):
        assert 0. < batch_size and batch_size <= 1.
        batch_size = int(batch_size * len(trainX))

    optimizer = optimizer(flow.parameters(), **optimizer_kwargs)

    train_losses, val_losses = [], []
    
    val_loss = np.inf
    best_loss = np.inf
    best_epoch = 0
    best_model = None

    if loss_f is None:
        loss_f = lambda batch, idx, cond=None: flow.nll(batch, cond=cond)

    best_model = TemporaryFile()

    try:
        with tqdm(n_epochs, leave=True, position=0) as tq:
            for epoch in range(1, n_epochs + 1):
                # Train
                flow.train()
                X = trainX
                idx = torch.randperm(len(X), device=X.device)
                for n in range(0, len(X), batch_size):
                    if len(X) - n == 1: continue
                    subidx = idx[n:n + batch_size]
                    batch = X[subidx].to(flow.device)
                    if conditional:
                        cond = cond_train[subidx].to(flow.device)

                    loss = loss_f(batch, subidx, cond=cond).mean()

                    assert not torch.isnan(loss) and not torch.isinf(loss)
                    
                    # Pytorch recipe: zero_grad - backward - step
                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    if gradient_clipping is not None:
                        assert gradient_clipping > 0
                        nn.utils.clip_grad_norm_(
                            flow.parameters(), 
                            gradient_clipping
                        )

                    optimizer.step()
                    
                    train_losses.append((epoch + n / len(trainX), loss.item()))

                    tq.set_postfix(OrderedDict(
                        epoch_progress='%.3d%%' % (n / len(X) * 100),
                        train_loss='%+.3e' % loss.item(), 
                        last_val_loss='%+.3e' % val_loss, 
                        best_epoch=best_epoch, 
                        best_loss='%+.3e' % best_loss
                    ))

                    if post_training_f is not None:
                        post_training_f(batch, subidx, cond=cond)
                
                # Validation
                flow.eval()
                X = valX
                idx = torch.randperm(len(X), device=X.device)
                with torch.no_grad(): # won't accumulate info about gradient
                    val_loss = 0.
                    for n in range(0, len(X), batch_size):
                        subidx = idx[n:n + batch_size]
                        batch = X[subidx].to(flow.device)
                        if conditional:
                            cond = cond_val[subidx].to(flow.device)

                        val_loss += (
                            loss_f(batch, subidx, cond=cond) / len(X)
                        ).sum().item()

                    val_losses.append((epoch, val_loss))

                    if post_validation_f is not None:
                        post_validation_f()

                assert not np.isnan(val_loss)# and not np.isinf(val_loss)

                # Early stopping
                if best_loss > val_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    
                    best_model.seek(0)
                    torch.save(flow.state_dict(), best_model)
                    
                tq.update()
                tq.set_postfix(OrderedDict(
                    epoch_progress='100%',
                    train_loss='%+.3e' % loss.item(), 
                    last_val_loss='%+.3e' % val_loss, 
                    best_epoch=best_epoch, 
                    best_loss='%+.3e' % best_loss
                ))

                if patience and epoch - best_epoch >= patience:
                    break

    except KeyboardInterrupt:
        print('Interrupted at epoch', epoch)
        pass # halt training without losing everything

    # Load best model before exiting
    best_model.seek(0)
    flow.load_state_dict(torch.load(best_model))
    best_model.close()

    flow.eval() # pass to eval mode before returning

    return train_losses, val_losses


def plot_losses(train_losses, val_losses, cellsize=(6, 4)):
    """Plot train and validation losses from a `train` call.

    Args:
        train_losses (list): (epoch, loss) pairs to plot for training.
        val_losses (list): (epoch, loss) pairs to plot for validation.
        cellsize (tuple): (width, height) for each cell in the plot.
    """

    best_epoch, best_loss = min(val_losses, key=lambda pair: pair[1])

    w, h = cellsize
    fig, axes = plt.subplots(1, 2, figsize=(w * 2, h * 1))

    axes[0].set_title('train_loss')
    axes[0].plot(*np.array(train_losses).T)

    axes[1].set_title('val_loss')
    axes[1].plot(*np.array(val_losses).T)
    axes[1].axvline(best_epoch, ls='dashed', color='gray')


def test_nll(flow, testX, cond_test=None, batch_size=32):
    """Compute test nll using batches.

    Args:
        flow (Flow): flow to train.
        testX (torch.Tensor): test dataset.
        cond_test (torch.Tensor or None): conditioning tensor for testX, 
            if the flow is conditional.
        batch_size (int or float): if float, ratio of testX to use per batch.
            If int, batch size.
    """

    assert isinstance(flow, Flow)
    assert flow.prior is not None, 'flow.prior is required'

    conditional = cond_test is not None

    if isinstance(batch_size, float):
        assert 0. < batch_size and batch_size <= 1.
        batch_size = int(batch_size * len(trainX))

    flow.eval()
    with torch.no_grad(): # won't accumulate info about gradient
        loss = 0.
        for n in range(0, len(testX), batch_size):
            idx = torch.arange(n, min(n + batch_size, len(testX)))

            batch = testX[idx].to(flow.device)
            if conditional:
                cond = cond_test[idx].to(flow.device)
            else:
                cond = None

            loss += (flow.nll(batch, cond=cond).sum() / len(testX)).item()

    return loss
