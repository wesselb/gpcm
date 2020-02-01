from collections import namedtuple

import lab as B
import numpy as np
from matrix import AbstractMatrix, Woodbury
from plum import Dispatcher

__all__ = ['summarise_samples',
           'estimate_psd',
           'invert_perm',
           'pd_inv',
           'collect',
           'autocorr',
           'method']

_dispatch = Dispatcher()


@B.matmul.extend(B.Numeric, B.Numeric, B.Numeric)
def matmul(a, b, c, tr_a=False, tr_b=False, tr_c=False):
    return B.mm(a, B.mm(b, c, tr_a=tr_b, tr_b=tr_c), tr_a=tr_a)


def summarise_samples(x, samples):
    """Summarise samples.

    Args:
        x (vector): Inputs of samples.
        samples (tensor): Samples, with the first dimension corresponding
            to different samples.

    Returns:
        :class:`collections.namedtuple`: Named tuple containing various
            statistics of the samples.
    """
    x, samples = B.to_numpy(x, samples)
    random_inds = np.random.permutation(B.shape(samples)[0])[:3]
    return collect(x=B.to_numpy(x),
                   mean=B.mean(samples, axis=0),
                   err_68_lower=np.percentile(samples, 32, axis=0),
                   err_68_upper=np.percentile(samples, 100 - 32, axis=0),
                   err_95_lower=np.percentile(samples, 2.5, axis=0),
                   err_95_upper=np.percentile(samples, 100 - 2.5, axis=0),
                   err_99_lower=np.percentile(samples, 0.15, axis=0),
                   err_99_upper=np.percentile(samples, 100 - 0.15, axis=0),
                   samples=B.transpose(samples)[..., random_inds])


def estimate_psd(t, k, n_zero=2_000, db=False):
    """Estimate the PSD from samples of the kernel.

    Args:
        t (vector): Time points of the kernel, which should be a linear space
            starting from the origin.
        k (vector): Kernel.
        n_zero (int, optional): Zero padding. Defaults to `2_000`.
        db (bool, optional): Convert to decibel. Defaults to `False`.

    Returns:
        vector: PSD, correctly scaled.
    """
    # Convert to NumPy for compatibility with frameworks.
    t, k = B.to_numpy(t, k)

    if t[0] != 0:
        raise ValueError('Time points must start at zero.')

    # Perform zero padding.
    k = B.concat(k, B.zeros(n_zero))

    # Symmetrise and Fourier transform.
    k_symmetric = B.concat(k, k[1:-1][::-1])
    psd = np.fft.fft(k_symmetric)
    freqs = np.fft.fftfreq(len(psd))/(t[1] - t[0])

    # Should be real and positive, but the numerics may not be in our favour.
    psd = np.abs(np.real(psd))

    # Now scale appropriately: the total power should equal `k[0]`.
    total_power = np.trapz(y=psd, x=freqs)
    psd /= total_power/k[0]

    # Convert to dB.
    if db:
        psd = 10*np.log10(psd)

    # Only return non-negative frequencies.
    inds = freqs >= 0
    freqs = freqs[inds]
    psd = psd[inds]

    return freqs, psd


def invert_perm(perm):
    """Compute the inverse of a permutation.

    Args:
        perm (list): Permutation to invert.
    """
    inverse_perm = np.array([-1]*len(perm), dtype=int)
    for i, p in enumerate(perm):
        inverse_perm[p] = i
    return inverse_perm


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


def autocorr(x, lags=None, normalise=True):
    """Estimate the autocorrelation.

    Args:
        x (vector): Time series to estimate autocorrelation of.
        lags (int, optional): Number of lags. Defaults to all lags.
        normalise (bool, optional): Normalise estimation. Defaults to `True`.

    Returns:
        vector: Autocorrelation.
    """
    # Convert to NumPy for compatibility with frameworks.
    x = B.to_numpy(x)

    # Compute autocorrelation.
    x = np.reshape(x, -1)  # Flatten the input.
    x = x - np.mean(x)
    k = np.correlate(x, x, mode='full')[:x.size][::-1]
    k /= np.arange(x.size, 0, -1)  # Divide by the number of estimates.

    # Get the right number of lags.
    if lags is not None:
        k = k[:lags + 1]

    # Normalise by the variance.
    if normalise:
        k = k/max(k)

    return k


def method(cls):
    """Decorator to add the function as a method to a class.

    Args:
        cls (type): Class to add the function as a method to.
    """

    def decorator(f):
        setattr(cls, f.__name__, f)
        return f

    return decorator
