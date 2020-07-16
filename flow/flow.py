"""
Abstract classes for the implementation of Flows.

* `Flow`: base abstract class for any flow.
* `Sequential`: flow as a composition of n subflows.
* `Conditioner`: abstract class for any flow defined 
    as the combination of a transformer and a conditioner.
* `Transformer`: abstract class for a transformer.

Also function:

* `inv_flow`: generates an inheriting class from a Flow 
    that swaps its _transform and _invert methods.
"""

import torch
from torch import nn

from .prior import Normal as NormalPrior


class Flow(nn.Module):
    r"""Base abstract class for any Flow. 

    A Flow represents a diffeomorphic T such that U = T(X),
    where X is the data distribution and U is the base distribution,
    a standard Normal distribution by default.

    Any class that inherits from Flow needs to implement:
    ```python
    def _transform(self, x, log_det=False, **kwargs): 
        # Transforms x into u. Used for training.
        ...

    def _invert(self, u, log_det=False, **kwargs): 
        # Transforms u into x. Used for sampling.
        ...

    def warm_start(self, x, **kwargs):
        # Warm start operation for the flow, if necessary.
        ...
    ```

    Both _transform and _invert may also return log|det J_T| if log_det is True.
    Otherwise, they only return the transformed tensor.

    Note that in training, using forward or backward KL divergence,
    _transform or _invert should be differentiable w.r.t. the flow's parameters,
    respectively. Otherwise, the flow would not learn.

    Remember that it's convenient for training stability to initialize
    the flow so that it performs the identity transformation (or close to it).
    """

    def __init__(self, dim=1, prior=None, **kwargs):
        """
        Args:
            dim (int): dimensionality of this flow. Defaults to 1.
            prior (class): prior class for U (inheriting `flow.prior.Prior`).
                Used for sampling and in the computation of nll.
                If None, defaults to `flow.prior.Normal`.
                If this flow is in a `Sequential`, its prior is ignored.
        """
        super().__init__()

        self.dim = dim

        if prior is None:
            prior = NormalPrior(dim=dim)
        else:
            assert prior.dim == dim
        
        self.prior = prior
        self.device = torch.device('cpu')


    def forward(self, t, invert=False, log_det=False, **kwargs):
        r"""Call _transform (x -> u) or _invert (u -> x) on t.

        Args:
            t (torch.Tensor): tensor to transform.
            invert (bool): whether to call _transform (True) 
                or _invert (False) on t.
            log_det (bool): whether to return \(\log |\det J_T|\)
                of the current transformation T.

        Returns:
            t: transformed tensor, either u if invert=False
                or x if invert=True.
            log_det: only returned if log_det=True.
                Tensor containing \(\log |\det J_T|\), 
                where T is the applied transformation. 
        """

        if not invert:
            return self._transform(t, **kwargs, log_det=log_det)
        else:
            return self._invert(t, **kwargs, log_det=log_det)


    # Override these methods
    def _transform(self, x, log_det=False, **kwargs):
        """Transform x into u."""
        raise NotImplementedError()

    def _invert(self, u, log_det=False, **kwargs):
        """Transform u into x."""
        raise NotImplementedError()

    def warm_start(self, x, **kwargs):
        """Perform a warm_start operation to the flow (optional).

        Args:
            x (torch.Tensor): dataset sample to use in the warming up.

        Returns:
            self
        """
        return self


    # Utilities
    def sample(self, n, **kwargs):
        """Generate n samples from X."""
        assert self.prior is not None
        
        u = self.prior.sample(n)
        x = self(u, **kwargs, invert=True)

        return x

    def nll(self, x, **kwargs):
        """Compute the negative log-likelihood of samples x.

        The result of this function can directly be used 
        as the MLE training loss for a flow.
        """
        assert self.prior is not None

        u, log_det = self(x, **kwargs, log_det=True)
        assert len(log_det.shape) == 1

        return self.prior.nll(u) - log_det

    def reverse_kl(self, n, nll_x, **kwargs):
        """Compute the reverse KL divergence of n prior samples.

        Used to train flows in reverse mode.
        Useful to create samplers from a known density function.

        Args:
            n (int): number of samples.
            nll_x (function): negative log-density function of x.
                Also receives **kwargs.
        """
        assert self.prior is not None

        u = self.prior.sample(n)
        loglk_u = -self.prior.nll(u)

        x, log_det = self(u, log_det=True, invert=True, **kwargs)
        assert len(log_det.shape) == 1

        loglk_x = -nll_x(x, **kwargs)

        return loglk_u - log_det - loglk_x


    # Device overrides
    def _update_device(self, device):
        """Update saved device for this flow and all its subcomponents."""
        if self.prior is not None:
            self.prior._update_device(device)

        self.device = device

    def to(self, device):
        """Override .to(device) so as to call _update_device(device)."""
        self._update_device(device)

        return super().to(device)

    def cpu(self):
        """Override .cpu so as to call .to method."""
        return self.to(torch.device('cpu'))

    def cuda(self):
        """Override .cuda so as to call .to method."""
        return self.to(torch.device('cuda', index=0))


def inv_flow(flow_cls, name=None):
    """Transform a Flow class so that _transform and _invert are swapped.
    
    Args:
        flow_cls (class): `Flow` class to inherit from.
        name (str): name to use for the new class. 
            If None, defaults to 'Inv' + flow_cls.__name__
    """
    if name is None:
        name = 'Inv' + flow_cls.__name__

    class InvFlow(flow_cls):

        # Extend forward to swap _transform and _invert
        def forward(self, *args, invert=False, **kwargs):
            return super().forward(*args, invert=not invert, **kwargs)

    InvFlow.__name__ = name
    InvFlow.__doc__ = (
        'Inverse flow. Note that _transform and _invert '
        'are swapped in this Flow.\n'
    ) + __doc__

    return InvFlow


class Sequential(Flow):
    """Flow defined by a sequence of flows.

    Note that flows are specified in the order X -> U.
    That means, for a Sequential Flow with steps T1, T2, T3: U = T3(T2(T1(X))).
    """

    def __init__(self, *flows, **kwargs):
        """
        Args:
            *flows (Flow): sequence of flows.
        """

        assert flows, 'Sequential constructor called with 0 flows'

        dim = { flow.dim for flow in flows }
        assert len(dim) == 1, \
            'All flows in a Sequential must have the same dim'
        dim = dim.pop() # just the one
        assert dim == kwargs.pop('dim', dim), 'dim and flows dim do not match'

        super().__init__(dim=dim, **kwargs)
        self.flows = nn.ModuleList(flows) # save flows in ModuleList


    # Method overrides
    def _transform(self, x, log_det=False, **kwargs):
        log_det_sum = torch.zeros(1, device=x.device)
        
        for flow in self.flows:
            res = flow(x, log_det=log_det, **kwargs)

            if log_det:
                x, log_det_i = res
                assert len(log_det_i.shape) == 1

                log_det_sum = log_det_sum + log_det_i
            else:
                x = res

        if log_det:
            return x, log_det_sum
        else:
            return x

    def _invert(self, u, log_det=False, **kwargs):
        log_det_sum = torch.zeros(1, device=u.device)

        for flow in reversed(self.flows):
            res = flow(u, invert=True, log_det=log_det, **kwargs)

            if log_det:
                u, log_det_i = res
                assert len(log_det_i.shape) == 1

                log_det_sum = log_det_sum + log_det_i
            else:
                u = res

        if log_det:
            return u, log_det_sum
        else:
            return u


    # Utilities
    def warm_start(self, x, **kwargs):
        """Call warm_start(x, **kwargs) to each subflow.

        Note that x will be progressively transformed before each 
        call to warm_start, from x to u.
        """

        for flow in self.flows:
            flow.warm_start(x, **kwargs)

            with torch.no_grad():
                x = flow(x, **kwargs)

        return self

    def __getitem__(self, k):
        """Access subflows by indexing. 

        Single ints return the corresponding subflow, 
        while slices return a `Sequential` of the corresponding subflows.
        """

        if isinstance(k, int):
            return self.flows[k]
        elif isinstance(k, slice):
            return Sequential(*self.flows[k])
        else:
            raise ValueError(k)

    def __iter__(self):
        for flow in self.flows:
            yield flow


    # Device overrides
    def _update_device(self, device):
        # Also call all its subflows _update_device(device) methods
        for flow in self.flows:
            flow._update_device(device)

        return super()._update_device(device)


class Conditioner(Flow):
    """Implement Flow by use of a conditioner and a transformer.

    This class is the conditioner itself, but acts as a flow,
    and receives the transformer as an input for its constructor. 

    Can also be used as a Conditional Flow, meaning, 
    a given input tensor conditions on the distribution modelled by the Flow.
    In that case, pass cond_dim > 0 to the constructor.
    If a Conditioner cannot be conditional, specify the class attribute,
    conditional = False. If a non-conditional Conditioner has cond_dim > 0,
    it raises a ValueError on initialization.

    Any class that inherits from Conditioner needs to implement:
    ```python
    def _h(self, x, cond=None, **kwargs): 
        # Return the (non-activated) tensor of parameters h 
        # corresponding to the given x. If this is a conditional flow,
        # the conditioning tensor is passed as the 'cond' kwarg.
        ...

    def _invert(self, u, cond=None, log_det=False, **kwargs): 
        # Transform u into x.
        ...
    ```

    Note that a `Conditioner` does not require an implementation 
    for method _transform, since it is dealt with by the transformer.
    However, it does need one for _invert, 
    since it depends on the implemented conditioner.

    Note that `Transformer` exposes a _h_init method that, 
    if it doesn't return None, should return an initialization value 
    for pre-activation parameters h. This is useful to initialize
    a Flow to the identity transformation, for training stability.
    For example, if your conditioner returns h as the result 
    of a feed-forward MLP, you could initialize all weights and biases 
    as randn() * 1e-3, and the last layer biases as the result of _h_init.
    """


    conditional = True
    """Whether this class can model conditional distributions (cond_dim > 0)."""

    def __init__(self, trnf, cond_dim=0, **kwargs):
        """
        Args:
            trnf (Transformer): transformer.
            cond_dim (int): if you want to use a conditional flow,
                this is the dimension of the conditioning tensor.
        """

        dim = trnf.dim
        assert kwargs.pop('dim', dim) == dim

        super().__init__(dim=dim, **kwargs)

        self.trnf = trnf
        assert cond_dim >= 0
        self.cond_dim = cond_dim

        # not conditional -> cond_dim == 0
        if not self.conditional and cond_dim:
            raise ValueError(
                'This Conditioner is non-conditional, '
                'so cond_dim needs to be 0.'
            )


    # Method overrides
    def forward(self, *args, cond=None, **kwargs):
        """Extend forward to include cond attribute, for conditional flows."""
        if self.conditional and self.cond_dim:
            if cond is None:
                raise ValueError('cond is None but cond_dim > 0')
            elif cond.size(1) != self.cond_dim:
                raise ValueError(
                    f'Invalid cond dim {cond.size(1)}; expected {self.cond_dim}'
                )
        elif cond is not None:
            raise ValueError('Passing cond != None to non-conditional flow')

        return super().forward(*args, cond=cond, **kwargs)

    def _transform(self, x, log_det=False, cond=None, **kwargs):
        h = self._h(x, cond=cond, **kwargs)
        
        return self.trnf(x, h, log_det=log_det, **kwargs)

    # Extend warm_start to also call trnf.warm_start
    def warm_start(self, x, **kwargs):
        self.trnf.warm_start(x, **kwargs)

        return super().warm_start(x, **kwargs)


    # Override these methods
    def _h(self, x, cond=None, **kwargs):
        """Compute the non-activated parameters for the transformer."""
        raise NotImplementedError()

    def _invert(self, u, log_det=False, cond=None, **kwargs):
        """Transform u into x."""
        raise NotImplementedError()


    # Utilities
    def _prepend_cond(self, x, cond=None):
        """Return torch.cat([cond, x], 1), broadcasting cond if necessary.
    
        If cond is None, does nothing to x. Useful to avoid checking for cond
        and preprocessing it every time.
        """
        if cond is None:
            return x
        else:
            if cond.size(0) < x.size(0):
                cond = cond.repeat(x.size(0) // cond.size(0), 1)
            
            assert cond.size(0) == x.size(0)
            return torch.cat([cond, x], 1)


    # Device overrides
    # Extend _update_device to also call it for its transformer.
    def _update_device(self, device):
        self.trnf._update_device(device)

        return super()._update_device(device)


class Transformer(Flow):
    """Transformer class used as a part of any Conditioner-Flow.

    Any class that inherits from Transformer needs to implement:
    ```python
    def __init__(self, **kwargs):
        # Extend the constructor to pass, to its base class,
        # the required kwarg h_dim >= 0. 
        ...
        super().__init__(h_dim=h_dim, **kwargs)
        ...

    def _activation(self, h, **kwargs): 
        # Transform h by activation before calling _transform or _invert.
        # 
        # For example, for a scale parameter, _activation could pass h
        # through a softplus function to make it positive.
        # Returns a tuple with the activated tensor parameters.
        ...

    def _transform(self, x, *h, log_det=False, **kwargs): 
        # Transform x into u using parameters h.
        ...

    def _invert(self, u, *h, log_det=False, **kwargs):
        # Transform u into x using parameters h.
        ...

    def _h_init(self):
        # Return initialization values for pre-activation h parameters.
        ...
    ```

    Note that forward, _transform and _invert all receive h,
    that contains the parameters for the transformation.

    The required attribute h_dim represents the dimensionality 
    of the required parameters for each dimension in the flow.
    As an example, an Affine transformer has h_dim=2 (loc and scale)
    for each dimension in the flow.

    CAUTION: all the three first methods need to be general enough 
        to work with dim=1 or an arbitrary dim,
        since the conditioner might pass any number of dimensions
        to transform depending on how it works.
    """

    def __init__(self, **kwargs):
        """""" # to avoid inheriting its parent in the documentation.
        super().__init__(**kwargs)

        self.h_dim = kwargs.get('h_dim', -1)
        assert self.h_dim >= 0

    def forward(self, t, h, invert=False, log_det=False, **kwargs):
        r"""Call _activation(h) and pass it to _transform or _invert.

        Args:
            t (torch.Tensor): tensor to transform.
            h (torch.Tensor): parameters for the transformation, 
                to be pre-processed with _activation. 
                This tensor comes from Conditioner._h.
            invert (bool): whether to call _transform (True) 
                or _invert (False) on t.
            log_det (bool): whether to return \(\log |\det J_T|\)
                of the current transformation T.

        Returns:
            t: transformed tensor, either u if invert=False
                or x if invert=True.
            log_det: only returned if log_det=True.
                Tensor containing \(\log |\det J_T|\), 
                where T is the applied transformation. 
        """

        h = self._activation(h, **kwargs)
        assert isinstance(h, tuple), 'trnf._activation(h) must return a tuple'

        if not invert: 
            return self._transform(t, *h, log_det=log_det, **kwargs)
        else:
            return self._invert(t, *h, log_det=log_det, **kwargs)


    # Override these methods
    def _activation(self, h, **kwargs):
        """Transform h by activation before calling _transform or _invert.

        Args:
            h (torch.Tensor): tensor with the pre-activation parameters.
        
        Returns:
            parameters: tuple of parameter tensors.

        Example:
            For a scale parameter, _activation could pass h
            through a softplus function to make it positive.
        """
        raise NotImplementedError()

    def _transform(self, x, *h, log_det=False, **kwargs):
        raise NotImplementedError()

    def _invert(self, u, *h, log_det=False, **kwargs):
        raise NotImplementedError()

    def _h_init(self):
        """Return initialization values for pre-activation h parameters.
        
        Returns:
            result: None if no initialization is required or
                tensor with shape (dim * h_dim,) with the initialization values.
        """
        return None # by default, no initialization suggested