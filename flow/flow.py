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

from .utils import Module


class UndefinedPriorException(Exception):
    
    def __init__(self, flow):
        self.flow = flow
        
    def __str__(self):
        return f'Undefined prior for {self.flow}'
    

class Flow(Module):
    r"""Base abstract class for any Flow. 

    A Flow represents a diffeomorphism T such that U = T(X),
    where X is the data distribution and U is the base distribution
    (i.e., a standard Normal distribution).

    Any class that inherits from Flow needs to implement:
    ```python
    def _activation(self, theta, **kwargs):
        # Transform theta parameters before calling _transform or _invert.
        # Returns a tuple with 0 or more activated parameters.
        ...
        
    def _transform(self, x, *theta, log_abs_det=False, **kwargs): 
        # Transform x into u. Used for training.
        ...

    def _invert(self, u, *theta, log_abs_det=False, **kwargs): 
        # Transform u into x. Used for sampling.
        ...
    ```
    
    Also, optionally, override these methods:
    ```python
    def _warm_start(self, x, *theta, **kwargs):
        # Warm start operation for the flow, if necessary.
        ...
        
    def _theta_init(self):
        # Return the pre-activation initialization value for the external parameters theta.
        # Shape must be (theta_dim,).
        ...
    ```
    
    Every flow must also define an `theta_dims` attribute,
    containing a list with the dimensionality of the external flow's parameters 
    for each dimension separately (they might be different).
    Then, every method that receives theta, receives a tensor with
    the first dimension's parameters in theta[:theta_dims[0]], 
    followed by the second dimension's in theta[theta_dims[0]:theta_dims[1]], etc.
    
    Note that any paramater that is independent to the input
    and can be stored as a nn.Parameter inside the flow is not considered.
    
    Along with the `theta_dims` attribute, there's a `theta_dim` attribute 
    which sums all `theta_dims`. You can set either of them to set the other.
        - Given `theta_dim`, then `theta_dims = (theta_dim // dim,) * dim`.
        - Given `theta_dims`, set `theta_dim = sum(theta_dims)`.
    
    Both _transform and _invert have a log_abs_det parameter.
    If log_abs_det is False, returns only the transformed tensor.
    Otherwise, returns the transformed tensor and log|det J_T|.

    Note that in training, using forward/backward KL divergence,
    _transform/_invert should be differentiable w.r.t. the flow's parameters,
    respectively. Otherwise, the flow would not learn.

    Remember that it's convenient for training stability to initialize
    the flow so that it performs the identity transformation (or close to it).
    This is done by providing the appropriate value in the `h_init` method.
    """
    
    @property
    def theta_dim(self):
        return self._theta_dim
    
    @theta_dim.setter
    def theta_dim(self, value):
        assert value % self.dim == 0
        
        self._theta_dim = value
        self._theta_dims = (value // self.dim,) * self.dim
        
    @property
    def theta_dims(self):
        return self._theta_dims
    
    @theta_dims.setter
    def theta_dims(self, value):
        if isinstance(value, int):
            value = (value,) * self.dim
            
        assert isinstance(value, (list, tuple)) and \
            all(isinstance(x, int) and x >= 0 for x in value) and \
            len(value) == self.dim
        
        self._theta_dims = tuple(value)
        self._theta_dim = sum(self._theta_dims)
        
    def __init__(self, dim=1, prior=None):
        """
        Args:
            dim (int): dimensionality of this flow. Defaults to 1.    
            prior (class): prior class for U (inheriting `flow.prior.Prior`).
                Used for sampling and in the computation of loglk.
                If None, no prior will be defined.
                Be mindful that your top-most flow should have a prior;
                otherwise, `sample`, `loglk` and `reverse_kl` will raise an AssertionError.
        """
        super().__init__()

        self.dim = dim
        self.theta_dim = 0 # by default, unless re-specified by the subclass
        
        if prior is not None:
            self.prior = prior(dim=dim)
        else:
            self.prior = None


    def forward(self, t, theta=None, invert=False, log_abs_det=False, **kwargs):
        r"""Call _transform (x -> u) or _invert (u -> x) on t.

        Args:
            t (torch.Tensor): tensor to transform.
            theta (torch.Tensor): theta parameter, if any. Can be None only if self.theta_dim == 0.
            invert (bool): whether to call _transform (True) or _invert (False) on t.
            log_abs_det (bool): whether to return \(\log |\det J_T|\)
                of the current transformation T.

        Returns:
            t: transformed tensor, either u if invert=False
                or x if invert=True.
            log_abs_det: only returned if log_abs_det=True.
                Tensor containing \(\log |\det J_T|\), 
                where T is the applied transformation. 
        """
        assert (not self.theta_dim or theta is not None) and \
            (theta is None or self.theta_dim), 'theta_dim > 0 iif. theta is not None.'
        
        theta = self._activation(theta, **kwargs)
        assert isinstance(theta, (list, tuple)), 'trnf._activation(h) must return a tuple'

        if not invert:
            return self._transform(t, *theta, log_abs_det=log_abs_det, **kwargs)
        else:
            return self._invert(t, *theta, log_abs_det=log_abs_det, **kwargs)
        
    def warm_start(self, x, theta=None, **kwargs):
        """Perform a warm_start operation to the flow (optional).

        Args:
            x (torch.Tensor): dataset sample to use in warming up.
            theta (torch.Tensor): external parameters or None.

        Returns:
            self
        """
        super().warm_start(x)
        
        theta = self._activation(theta, **kwargs)
        self._warm_start(x, *theta, **kwargs)
        
        return self

    # Override these methods
    def _activation(self, theta, **kwargs):
        """Transform theta by activation before calling _transform or _invert.

        Args:
            theta (torch.Tensor): tensor with the pre-activation parameters.
        
        Returns:
            parameters: tuple of parameter tensors.

        Example:
            For a scale parameter, _activation could pass theta
            through a softplus function to make it positive.
        """
        return (theta,) if theta is not None else tuple()
        
    def _transform(self, x, *theta, log_abs_det=False, **kwargs):
        """Transform x into u."""
        raise NotImplementedError()

    def _invert(self, u, *theta, log_abs_det=False, **kwargs):
        """Transform u into x."""
        raise NotImplementedError()
        
    def _theta_init(self):
        """Return initialization values for pre-activation theta parameters.
        
        Returns:
            result: None if no initialization is required or
                tensor with shape (theta_dim,) with the initialization values.
        """
        return None # by default, no initialization suggested
    
    def _warm_start(self, x, *theta, **kwargs):
        return # does nothing by default

    # Utilities
    def sample(self, n, *args, **kwargs):
        """Generate n samples from X."""
        if self.prior is None: raise UndefinedPriorException(self)
        
        u = self.prior.sample(n)
        x = self(u, *args, **kwargs, invert=True, log_abs_det=False)

        return x

    def loglk(self, x, *args, **kwargs):
        """Compute the log-likelihood of samples x.
        """
        if self.prior is None: raise UndefinedPriorException(self)

        u, log_abs_det = self(x, *args, **kwargs, invert=False, log_abs_det=True)
        assert log_abs_det.shape == (x.size(0),)

        return self.prior.loglk(u) + log_abs_det
    
    def nll(self, *args, **kwargs):
        """Compute the *negative* log-likelihood of samples x.
        
        The result of this function can directly be used 
        as the MLE training loss for a flow.
        
        Refer to `loglk` for details about the function.
        """
        return -self.loglk(*args, **kwargs)

    def reverse_kl(self, n, loglk_f, *args, **kwargs):
        """Compute the reverse KL divergence of n prior samples.

        Used to train flows in reverse mode.
        Useful to create samplers from a known density function.

        Args:
            n (int): number of samples.
            loglk_f (function): log-density function(x, **kwargs) for x.
        """
        if self.prior is None: raise UndefinedPriorException(self)

        u = self.prior.sample(n)
        loglk_u = self.prior.loglk(u)

        x, log_abs_det = self(u, *args, log_abs_det=True, invert=True, **kwargs)
        assert log_abs_det.shape == (n,)

        loglk_x = loglk_f(x, **kwargs)
        assert loglk_x.shape == (n,)

        return loglk_u - log_abs_det - loglk_x


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
        'are swapped when calling this Flow.\n'
    ) + __doc__

    return InvFlow


class Conditioner(Flow):
    """Implement Flow by use of a Conditioner and a Transformer.

    This class is the conditioner itself, but acts as a flow,
    and receives the transformer as an input for its constructor. 
    The Transformer can be any other Flow that has theta_dim > 0.
    The Conditioner is the one responsible for providing that external theta.

    Can also be used as a Conditional Flow, meaning, 
    a given input tensor conditions on the distribution modelled by the Flow.
    In that case, pass cond_dim > 0 to the constructor.
    If a Conditioner cannot be conditional, specify the class attribute,
    conditional = False. If a non-conditional Conditioner has cond_dim > 0,
    it raises a ValueError on initialization.

    Any class that inherits from Conditioner needs to implement:
    ```python
    def _theta(self, x, cond=None, **kwargs): 
        # Return the (non-activated) tensor of parameters theta 
        # corresponding to the given x. If this is a conditional flow,
        # the conditioning tensor is passed as the 'cond' kwarg.
        ...

    def _invert(self, u, cond=None, log_abs_det=False, **kwargs): 
        # Transform u into x.
        ...
    ```

    Note that a `Conditioner` does not require an implementation 
    for its method _transform, since it is dealt with by the transformer.
    However, it does need one for _invert, 
    since it depends on the implemented conditioner.

    Note that the transformer has a _theta_init method that, 
    if it doesn't return None, should return an initialization value 
    for pre-activation parameters theta. This is useful to initialize
    a Flow to the identity transformation, for training stability.
    For example, if your conditioner returns theta as the result 
    of a feed-forward MLP, you could initialize all weights and biases 
    as randn() * 1e-3, and the last layer biases as the result of _theta_init.
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
        self.theta_dim = 0 # as the conditioner takes care of trnf.theta_dim
        
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
        # Extend forward to include asserts on the cond attribute
        if self.cond_dim:
            if cond is None:
                raise ValueError('cond is None but cond_dim > 0')
            elif cond.size(1) != self.cond_dim:
                raise ValueError(
                    f'Invalid cond dim {cond.size(1)}; expected {self.cond_dim}'
                )

        return super().forward(*args, cond=cond, **kwargs)

    def _transform(self, x, *theta, cond=None, log_abs_det=False, **kwargs):
        assert not theta
        theta = self._theta(x, cond=cond, **kwargs)
        
        return self.trnf(x, theta=theta, log_abs_det=log_abs_det, **kwargs)

    # Extend warm_start to also call trnf.warm_start
    def _warm_start(self, x, cond=None, **kwargs):
        super()._warm_start(x, cond=cond, **kwargs)
    
        theta = self._theta(x, cond=cond, **kwargs)
        self.trnf.warm_start(x, theta=theta, **kwargs)

    # Override these methods
    def _theta(self, x, cond=None, **kwargs):
        """Compute the non-activated parameters for the transformer."""
        raise NotImplementedError()

    def _invert(self, u, cond=None, log_abs_det=False, **kwargs):
        """Transform u into x."""
        raise NotImplementedError()


class Sequential(Flow):
    """Flow defined by a sequence of flows.

    Note that flows are specified in the order X -> U.
    That means, for a Sequential Flow with steps T1, T2, T3: U = T3(T2(T1(X))).
    
    If any internal flow has theta_dim > 0, this Flow inherits it.
    A Conditioner will be needed to get the values of all external parameters.
    
    As an example,
    ```python
    flow = Sequential(Linear, Linear, Constant(Linear(dim=3)), dim=3)
    ```
    
    Note that you can pass instanced flows (`Constant(Linear(dim=3))`)
    or Flow classes / functions(dim=1) that return flows. 
    If an entry parameter f in *flows is not instanced, 
    it will be called like f(dim=dim) and expected to return a Flow with that dimension.
    
    In the example, Linear has a mu and sigma parameter for each dimension.
    Hence, theta_dim of the Sequential will be 2 * 3 + 2 * 3 = 12, 
    with the first 6 dimensions passed to the first flow, the remaining to the other.
    """

    def __init__(self, *flows, dim=1, **kwargs):
        """
        Args:
            *flows (Flow): sequence of flows.
            dim (int): dimensionality of the flow.
        """
        
        assert flows, 'At least one subflow must be provided.'
        
        super().__init__(dim=dim, **kwargs)
        
        self.flows = nn.ModuleList([ f if isinstance(f, Flow) else f(dim=dim) for f in flows ])
        assert all(f.dim == self.dim for f in self.flows)
        
        self.theta_dims = tuple(
            sum( f.theta_dims[i] for f in self.flows )
            for i in range(self.dim)
        )

    # Device overrides
    def _update_device(self, device):
        super()._update_device(device)
        
        for flow in self.flows:
            flow.device = device

    # Method overrides
    def _activation(self, theta, **kwargs):
        assert not self.theta_dim or (theta is not None and theta.size(1) == self.theta_dim)
        
        if theta is not None:
            index = []
            s = 0
            for i in self.theta_dims:
                index.append(s)
                s += i
            
            res = []
            for flow in self.flows:
                theta2 = []
                
                for k in range(self.dim):
                    i = index[k]
                    j = i + flow.theta_dims[k]
                    
                    if j - i:
                        theta2.append(theta[:, i:j])
                        index[k] = j
                        
                if theta2:
                    res.append(torch.cat(theta2, dim=1))
                else:
                    res.append(None)
        else:
            res = [None] * len(self.flows)
            
        assert len(res) == len(self.flows) and all(
            theta.size(1) == flow.theta_dim
            for theta, flow in zip(res, self.flows)
            if theta is not None
        )
        
        return tuple(res)
        
    def _transform(self, *args, **kwargs):
        return self._forward(*args, **kwargs, invert=False)
    
    def _invert(self, *args, **kwargs):
        return self._forward(*args, **kwargs, invert=True)
    
    def _forward(self, x, *theta, log_abs_det=False, invert=False, **kwargs):
        log_abs_det_sum = 0.
        for n, flow in (
            enumerate(self.flows) if not invert else
            zip(range(len(self) - 1, -1, -1), reversed(self.flows))
        ):
            res = flow(x, theta=theta[n], invert=invert, log_abs_det=log_abs_det, **kwargs)

            if log_abs_det:
                x, log_abs_det_i = res
                assert len(log_abs_det_i.shape) == 1

                log_abs_det_sum = log_abs_det_sum + log_abs_det_i
            else:
                x = res

        if log_abs_det:
            return x, log_abs_det_sum
        else:
            return x
        
    def _theta_init(self):
        """Return initialization values for pre-activation theta parameters.
        
        Returns:
            result: None if no initialization is required or
                tensor with shape (theta_dim,) with the initialization values.
        """
        
        init = [[] for _ in range(self.dim)]
        
        for flow in self.flows:
            theta_init = flow._theta_init()
            if theta_init is not None:
                i = 0
                for k, d in enumerate(flow.theta_dims):
                    j = i + d
                    assert j <= theta_init.size(0)
                    if j - i:
                        init[k].append(theta_init[i:j])
                        i = j
        
        if any(l for l in init):
            return torch.cat([
                torch.cat(l, 0)
                for l in init
                if l
            ], 0)
        else:
            return None
    
    # Utilities    
    def _warm_start(self, x, *theta, **kwargs):
        """Call warm_start(x, **kwargs) to each subflow.

        Note that x will be progressively transformed before each 
        call to warm_start, from x to u.
        """
        super()._warm_start(x, *theta, **kwargs)
        
        for n, flow in enumerate(self.flows):
            flow.warm_start(x, theta=theta[n], **kwargs)

            with torch.no_grad():
                x = flow(x, theta=theta[n], **kwargs)

    def __getitem__(self, k):
        """Access subflows by indexing. 

        Single ints return the corresponding subflow, 
        while slices return a `Sequential` of the corresponding subflows.
        """

        if isinstance(k, int):
            return self.flows[k]
        else:
            raise ValueError(k)

    def __iter__(self):
        return iter(self.flows)
            
    def __len__(self):
        return len(self.flows)
    
    
# TODO: Remove this   
# def dec_trnf_1d(func):
#     """Decorator used to define a _transform/_invert method as a 1D operation.
#     If multiple dimensions are passed, the method will be called individually
#     per dimension and the results will be aggregated by this decorator."""
    
#     from functools import wraps
    
#     @wraps(func)
#     def f(self, x, *args, log_abs_det=False, **kwargs):
#         t = []
#         log_abs_det_sum = 0
        
#         for d in range(x.size(1)):
#             res = func(
#                 self, x[:, [d]],
#                 # all args are assumed to be tensors
#                 *(a[:, [d]] for a in args), 
#                 log_abs_det=log_abs_det,
#                 **kwargs
#             )
            
#             if log_abs_det:
#                 t_i, log_abs_det_i = res
#                 log_abs_det_sum += log_abs_det_i
#             else:
#                 t_i = res
                
#             t.append(t_i)
            
#         t = torch.cat(t, 1)
#         if log_abs_det:
#             return t, log_abs_det_sum
#         else:
#             return t
        
#     return f