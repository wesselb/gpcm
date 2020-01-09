from collections import namedtuple
import numpy as np

__all__ = ['collect', 'autocorr']


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
