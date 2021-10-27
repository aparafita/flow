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

    A Flow represents a diffeomorphic T such that U = T(X),
    where X is the data distribution and U is the base distribution,
    a standard Normal distribution by default.

    Any class that inherits from Flow needs to implement:
    ```python
    def _activation(self, h, **kwargs):
        # Transform h by activation before calling _transform or _invert.
        # Returns a tuple with 0 or more activated parameters.
        ...
        
    def _transform(self, x, *h, log_det=False, **kwargs): 
        # Transform x into u. Used for training.
        ...

    def _invert(self, u, *h, log_det=False, **kwargs): 
        # Transform u into x. Used for sampling.
        ...
    ```
    
    Also, optionally, override these methods:
    ```python
    def warm_start(self, x):
        # Warm start operation for the flow, if necessary.
        ...
        
    def h_init(self):
        # Return the pre-activation initialization value for the external parameters h.
        # Shape must be (h_dim,).
        ...
    ```
    
    Every flow must also define an `h_dim` attribute,
    containing the total dimensionality of the external flow's parameters.
    By this, it means that any paramater that is independent to the input
    and can be stored as a nn.Parameter inside the flow is not considered.
    External parameters come as a 'h' parameter of each function.
    `h_dim` is then the total dimensionality of this h parameter.
    `h_dim` can be 0 or a positive integer (0 by default, if it is not defined). 
    Alternatively, you can set the `h_dim_1d` attribute 
    to refer to the number of parameters per-dimension. 
    This sets h_dim automatically as self.h_dim = self.h_dim_1d * self.dim.
    
    Both _transform and _invert may also return log|det J_T| if log_det is True.
    Otherwise, they only return the transformed tensor.

    Note that in training, using forward or backward KL divergence,
    _transform or _invert should be differentiable w.r.t. the flow's parameters,
    respectively. Otherwise, the flow would not learn.

    Remember that it's convenient for training stability to initialize
    the flow so that it performs the identity transformation (or close to it).
    This is done by providing the appropriate value in the `h_init` method.
    """
    
    @property
    def h_dim_1d(self):
        return self.h_dim // self.dim
    
    @h_dim_1d.setter
    def h_dim_1d(self, value):
        self.h_dim = value * self.dim

        
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
        self.h_dim = 0 # by default, unless re-specified by the subclass
        
        if prior is not None:
            self.prior = prior(dim=dim)
        else:
            self.prior = None


    def forward(self, t, h=None, invert=False, log_det=False, **kwargs):
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
        assert not self.h_dim or h is not None, 'If h_dim > 0, then h should be passed.'
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
        return (h,) if h is not None else tuple()
        
    def _transform(self, x, h=None, log_det=False, **kwargs):
        """Transform x into u."""
        raise NotImplementedError()

    def _invert(self, u, h=None, log_det=False, **kwargs):
        """Transform u into x."""
        raise NotImplementedError()
        
    def _h_init(self):
        """Return initialization values for pre-activation h parameters.
        
        Returns:
            result: None if no initialization is required or
                tensor with shape (h_dim,) with the initialization values.
        """
        return None # by default, no initialization suggested

    def warm_start(self, x, h=None, **kwargs):
        """Perform a warm_start operation to the flow (optional).

        Args:
            x (torch.Tensor): dataset sample to use in warming up.
            h (torch.Tensor): external parameters or None.

        Returns:
            self
        """
        return super().warm_start()


    # Utilities
    def sample(self, n, **kwargs):
        """Generate n samples from X."""
        if self.prior is None: 
            raise UndefinedPriorException(self)
        
        u = self.prior.sample(n)
        x = self(u, **kwargs, invert=True)

        return x

    def loglk(self, x, **kwargs):
        """Compute the log-likelihood of samples x.
        """
        if self.prior is None: 
            raise UndefinedPriorException(self)

        u, log_det = self(x, **kwargs, log_det=True)
        assert len(log_det.shape) == 1

        return self.prior.loglk(u) + log_det
    
    def nll(self, *args, **kwargs):
        """Compute the *negative* log-likelihood of samples x.
        
        The result of this function can directly be used 
        as the MLE training loss for a flow.
        
        Refer to `loglk` for details about the function.
        """
        return -self.loglk(*args, **kwargs)

    def reverse_kl(self, n, loglk_f, **kwargs):
        """Compute the reverse KL divergence of n prior samples.

        Used to train flows in reverse mode.
        Useful to create samplers from a known density function.

        Args:
            n (int): number of samples.
            loglk_f (function): log-density function(x, **kwargs) for x.
        """
        if self.prior is None: 
            raise UndefinedPriorException(self)

        u = self.prior.sample(n)
        loglk_u = self.prior.loglk(u)

        x, log_det = self(u, log_det=True, invert=True, **kwargs)
        assert log_det.shape == (n,)

        loglk_x = loglk_f(x, **kwargs)
        assert loglk_x.shape == (n,)

        return loglk_u - log_det - loglk_x


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
    for its method _transform, since it is dealt with by the transformer.
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
        self.h_dim = 0 # as the conditioner takes care of trnf.h_dim
        
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
        elif cond is not None:
            raise ValueError('Passing cond != None to non-conditional flow')

        return super().forward(*args, cond=cond, **kwargs)

    def _transform(self, x, *h, log_det=False, cond=None, **kwargs):
        assert not h
        h = self._h(x, cond=cond, **kwargs)
        
        return self.trnf(x, h=h, log_det=log_det, **kwargs)

    # Extend warm_start to also call trnf.warm_start
    def warm_start(self, x, h=None, cond=None, **kwargs):
        super().warm_start(x, h=h, cond=cond, **kwargs)
    
        assert h is None
        h = self._h(x, cond=cond, **kwargs)
        self.trnf.warm_start(x, h=h, **kwargs)

        return self

    # Override these methods
    def _h(self, x, cond=None, **kwargs):
        """Compute the non-activated parameters for the transformer."""
        raise NotImplementedError()

    def _invert(self, u, *h, log_det=False, cond=None, **kwargs):
        """Transform u into x."""
        assert not h
        raise NotImplementedError()


class Sequential(Flow):
    """Flow defined by a sequence of flows.

    Note that flows are specified in the order X -> U.
    That means, for a Sequential Flow with steps T1, T2, T3: U = T3(T2(T1(X))).
    """

    def __init__(self, *flow_cls, dim=1, **kwargs):
        """
        Args:
            *flows (Flow): sequence of flows.
        """
        pass
        """
        assert flows, 'Sequential constructor called with 0 flows'

        dim = { flow.dim for flow in flows }
        assert len(dim) == 1, \
            'All flows in a Sequential must have the same dim'
        dim = dim.pop() # just the one
        assert dim == kwargs.pop('dim', dim), 'dim and flows dim do not match'
        
        super().__init__(dim=dim, **kwargs)
        
        self.flows = nn.ModuleList(flows) # save flows in ModuleList
        """
        
        assert flow_cls
        
        super().__init__(dim=dim, **kwargs)
        
        self.flows = nn.ModuleList([ cls(dim=dim) for cls in flow_cls ])
        
        self.h_dims = [ f.h_dim for f in self.flows ]
        self.h_dim = sum(self.h_dims)        

    # Device overrides
    def _update_device(self, device):
        super()._update_device(device)
        
        for flow in self.flows:
            flow.device = device

    # Method overrides
    def _activation(self, h, **kwargs):
        assert not self.h_dim or (h is not None and h.size(1) == self.h_dim)
        
        if h is not None:
            h = h.view(h.size(0), self.dim, -1)
            res = []
            i = 0
            for flow in self.flows:
                j = i + flow.h_dim_1d
                
                if j - i:
                    res.append(h[..., i:j].flatten(1))
                else:
                    res.append(None)

                i = j
        else:
            res = [None] * len(self)
        
        return tuple(res)
        
    def _transform(self, *args, **kwargs):
        return self._forward(*args, **kwargs, invert=False)
    
    def _invert(self, *args, **kwargs):
        return self._forward(*args, **kwargs, invert=True)
    
    def _forward(self, x, *h, log_det=False, invert=False, **kwargs):
        log_det_sum = torch.zeros(1, device=x.device)
        it = zip(range(len(self) - 1, -1, -1), reversed(self.flows)) if invert else enumerate(self.flows)
        for n, flow in it:
            res = flow(x, h=h[n], invert=invert, log_det=log_det, **kwargs)

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
        
    def _h_init(self):
        """Return initialization values for pre-activation h parameters.
        
        Returns:
            result: None if no initialization is required or
                tensor with shape (h_dim,) with the initialization values.
        """
        return torch.cat([
            init.view(self.dim, -1)
            for init in (f._h_init() for f in self.flows)
            if init is not None
        ], 1).flatten()
    
    # Utilities
    def warm_start(self, x, h=None, **kwargs):
        """Call warm_start(x, **kwargs) to each subflow.

        Note that x will be progressively transformed before each 
        call to warm_start, from x to u.
        """
        h = self._activation(h, **kwargs)

        for n, flow in enumerate(self.flows):
            flow.warm_start(x, h=h[n], **kwargs)

            with torch.no_grad():
                x = flow(x, h=h[n], **kwargs)

        return super().warm_start(x, h=h, **kwargs)

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
            
    def __len__(self):
        return len(self.flows)
        
        
class ConditionerNet(Module):
    """Conditioner parameters network.

    Used to compute the parameters h passed to a transformer.
    If its input_dim is 0 (i.e., a Flow of dimension 1 without conditioning),
    returns a learnable tensor containing the required result.
    """

    def __init__(
        self, 
        input_dim, output_dim, h_init=None, net_f=None
    ):
        """
        Args:
            input_dim (int): input dimensionality.
            output_dim (int): output dimensionality. 
                Total dimension of all parameters combined.
            h_init (torch.Tensor): tensor of shape (output_dim,) to use as initializer.
                If None, original initialization used.
            net_f (function): function net_f(input_dim, output_dim, init=None)
                that creates a network with the given input and output dimensions,
                and possibly receives an init Tensor that,
                when not None, indicates that the function should be set 
                so that it always return that tensor. 
                This helps in initializing a tensor to the identity.
        """
        assert net_f is not None
        
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        if h_init is not None:
            assert h_init.shape == (output_dim,)
            
        if input_dim:
            self.net = net_f(input_dim, output_dim, init=h_init)
        else:
            if h_init is None:
                h_init = torch.randn(output_dim)

            self.parameter = nn.Parameter(h_init.unsqueeze(0))

    def forward(self, x):
        """Feed-forward pass."""
        if self.input_dim:
            return self.net(x)
        else:
            return self.parameter.repeat(x.size(0), 1)

    def warm_start(self, x, **kwargs):
        super().warm_start(x, **kwargs)
        
        if self.input_dim:
            bn = self.net[0]
            assert bn.__class__.__name__.startswith('BatchNorm')

            bn.running_mean.data = x.mean(0)
            bn.running_var.data = x.var(0)
        
        return self
    
'''    
class Sequential1D(Sequential):
    """Flow defined by a sequence of 1D-flows.
    
    This makes it possible to define a single conditioner for all subflows,
    helping in mitigating overfitting.
    """

    def __init__(self, *flows, cond_dim=0, net_f=None, **kwargs):
        """
        Args:
            *flows (Flow): sequence of flows.
            dim (int): dimensions of the flow.
            cond_dim (int): dimensions of the conditioning input.
            net_f (function): function(input_dim, output_dim, init=None).
                input_dim will actually be the passed cond_dim.
                Note that the actual network will be a `ConditionerNet`,
                which admits cond_dim == 0, in which case a single parameter
                will be trained instead of the network returned by net_f.
        """
        
        assert kwargs.pop('dim', 1) == 1 # only works with 1D flows
        assert net_f is not None

        super().__init__(*flows, dim=1, **kwargs)
        
        self.cond_dim = cond_dim
        self.h_dims = [ getattr(flow, 'h_dim', 0) for flow in self ]
        self.h_dim = sum(self.h_dims)
        
        init = [
            m()
            for m in (getattr(flow, '_h_init', None) for flow in self) 
            if m is not None
        ]
        
        if init: init = torch.cat(init, -1)
        else: init = None
        
        self.net = ConditionerNet(self.cond_dim, self.h_dim, h_init=init, net_f=net_f)

    # Method overrides
    def _create_n_kwargs(self, x, kwargs):
        # Compute h once, and set it as layer-specific kwargs
        cond = kwargs.pop('cond', None)
        if cond is None:
            cond = x[:, :0] # to get the appropriate size(0)
            
        h = self.net(cond)
        
        i = 0
        for n, h_dim in enumerate(self.h_dims):
            if h_dim: 
                kwargs[f'__{n}_h'] = h[:, i:i + h_dim]
                i += h_dim
            
    def forward(self, x, invert=False, log_det=False, **kwargs):
        self._create_n_kwargs(x, kwargs)
        
        return super().forward(x, invert=invert, log_det=log_det, **kwargs)
        
    def warm_start(self, x, **kwargs):
        self._create_n_kwargs(x, kwargs)
        
        return super().warm_start(x, **kwargs)

    # Utilities
    def __getitem__(self, k):
        """Access subflows by indexing. 

        Single ints return the corresponding subflow, 
        while slices return a `Sequential` of the corresponding subflows.
        """
        
        if isinstance(k, slice):
            raise ValueError("Can't slice a Sequential1D") 
            # since we would need to slice the network
        else:
            return super().__getitem__(k)
'''

def dec_trnf_1d(func):
    """Decorator used to define a _transform/_invert method as a 1D operation.
    If multiple dimensions are passed, the method will be called individually
    per dimension and the results will be aggregated by this decorator."""
    
    from functools import wraps
    
    @wraps(func)
    def f(self, x, *args, **kwargs):
        t = []
        
        log_det = kwargs.get('log_det', False)
        log_det_sum = 0
        
        for d in range(x.size(1)):
            res = func(
                self, x[:, [d]],
                # all args are assumed to be tensors
                *(a[:, [d]] for a in args), 
                **kwargs
            )
            
            if log_det:
                t_i, log_det_i = res
                log_det_sum += log_det_i
            else:
                t_i = res
                
            t.append(t_i)
            
        t = torch.cat(t, 1)
        if log_det:
            return t, log_det_sum
        else:
            return t
        
    return f

'''
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
        #
        # Must return a tuple with the activated tensor parameters (even if it's just one).
        ...

    def _transform(self, x, *h, log_det=False, **kwargs): 
        # Transform x into u using parameters h.
        ...

    def _invert(self, u, *h, log_det=False, **kwargs):
        # Transform u into x using parameters h.
        ...
    ```
    Also, any subclass must define their `h_dim` attribute.
    As an example, an Affine transformer has h_dim=2 * self.dim 
    (loc and scale, one for each dimension).

    Note that forward, _transform and _invert all receive h,
    that contains the parameters for the transformation.

    CAUTION: all the three first methods need to be general enough 
        to work with dim=1 or an arbitrary dim,
        since the conditioner might pass any number of dimensions
        to transform depending on how it works.
    """

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
                tensor with shape (h_dim,) with the initialization values.
        """
        return None # by default, no initialization suggested
'''
Transformer = Flow