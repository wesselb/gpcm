from collections import namedtuple

import lab as B
import numpy as np
from matrix import AbstractMatrix, Woodbury
from plum import Dispatcher

__all__ = ['pd_inv', 'collect', 'autocorr', 'method']

_dispatch = Dispatcher()


@_dispatch({B.Numeric, AbstractMatrix})
def pd_inv(a):
    """Invert `a` using that `a` is positve definite.

    Args:
        a (matrix): Matrix to invert.

    Return:
        matrix: Inverse of `a`.
    """
    return B.cholsolve(B.chol(a), B.eye(a))


@_dispatch(Woodbury)
def pd_inv(a):
    # In this case there is no need to use the Cholesky decomposition.
    return B.inv(a)


def collect(name='Quantities', **kw_args):
    """Construct a named tuple with certain attributes.

    Args:
        **kw_args (object): Keyword argument specifying the attributes.
        name (str): Name of the named tuple. Defaults to "Quantities".

    Returns:
        :class:`collections.namedtuple`: Named tuple of the model with the
            specified attributes.
    """
    return namedtuple(name, kw_args)(**kw_args)


def autocorr(x, lags):
    """Estimate the autocorrelation.

    Args:
        x (vector): Time series to estimate autocorrelation of.
        lags (int): Number of lags.

    Returns:
        vector: Autocorrelation.
    """
    if lags < 0:
        raise ValueError('The number of lags must be positive.')
    x = np.reshape(x, -1)  # Flatten the input.
    x = x - np.mean(x)
    k = np.correlate(x, x, mode='full')[:x.size][::-1]
    k /= np.arange(x.size, 0, -1)  # Divide by the number of estimates.
    return k[:lags + 1]


def method(cls):
    """Decorator to add the function as a method to a class.

    Args:
        cls (type): Class to add the function as a method to.
    """
    def decorator(f):
        setattr(cls, f.__name__, f)
        return f

    return decorator
