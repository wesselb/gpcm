from collections import namedtuple
from varz import Vars, minimise_l_bfgs_b

import jax
import lab as B
import numpy as np
from plum import Dispatcher
from stheno import Normal
from scipy.signal import hilbert

__all__ = [
    "min_phase",
    "summarise_samples",
    "estimate_psd",
    "invert_perm",
    "closest_psd",
    "collect",
    "autocorr",
    "method",
    "hessian",
    "maximum_a_posteriori",
    "laplace_approximation",
]

_dispatch = Dispatcher()


@B.matmul.dispatch
def matmul(
    a: B.Numeric,
    b: B.Numeric,
    c: B.Numeric,
    tr_a=False,
    tr_b=False,
    tr_c=False,
):
    return B.mm(a, B.mm(b, c, tr_a=tr_b, tr_b=tr_c), tr_a=tr_a)


def min_phase(h):
    """Minimum phase transform using the Hilbert transform.

    Args:
        h (vector): Filter to transform.

    Returns:
        vector: Minimum phase filter version of `h`.
    """
    h = B.to_numpy(h)
    spec = np.fft.fft(h)
    phase = np.imag(-hilbert(np.log(np.abs(spec))))
    return np.real(np.fft.ifft(np.abs(spec) * np.exp(1j * phase)))


def summarise_samples(x, samples, db=False):
    """Summarise samples.

    Args:
        x (vector): Inputs of samples.
        samples (tensor): Samples, with the first dimension corresponding
            to different samples.
        db (bool, optional): Convert to decibels.

    Returns:
        :class:`collections.namedtuple`: Named tuple containing various
            statistics of the samples.
    """
    x, samples = B.to_numpy(x, samples)
    random_inds = np.random.permutation(B.shape(samples)[0])[:3]

    def transform(x):
        if db:
            return 10 * np.log10(x)
        else:
            return x

    perm = tuple(reversed(range(B.rank(samples))))  # Reverse all dimensions.
    return collect(
        x=B.to_numpy(x),
        mean=transform(B.mean(samples, axis=0)),
        var=transform(B.std(samples, axis=0)) ** 2,
        err_68_lower=transform(B.quantile(samples, 0.32, axis=0)),
        err_68_upper=transform(B.quantile(samples, 1 - 0.32, axis=0)),
        err_95_lower=transform(B.quantile(samples, 0.025, axis=0)),
        err_95_upper=transform(B.quantile(samples, 1 - 0.025, axis=0)),
        err_99_lower=transform(B.quantile(samples, 0.0015, axis=0)),
        err_99_upper=transform(B.quantile(samples, 1 - 0.0015, axis=0)),
        samples=transform(B.transpose(samples, perm=perm)[..., random_inds]),
        all_samples=transform(B.transpose(samples, perm=perm)),
    )


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
        raise ValueError("Time points must start at zero.")

    # Perform zero padding.
    k = B.concat(k, B.zeros(n_zero))

    # Symmetrise and Fourier transform.
    k_symmetric = B.concat(k, k[1:-1][::-1])
    psd = np.fft.fft(k_symmetric)
    freqs = np.fft.fftfreq(len(psd)) / (t[1] - t[0])

    # Should be real and positive, but the numerics may not be in our favour.
    psd = np.abs(np.real(psd))

    # Now scale appropriately: the total power should equal `k[0]`.
    total_power = np.trapz(y=psd, x=freqs)
    psd /= total_power / k[0]

    # Convert to dB.
    if db:
        psd = 10 * np.log10(psd)

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
    inverse_perm = np.array([-1] * len(perm), dtype=int)
    for i, p in enumerate(perm):
        inverse_perm[p] = i
    return inverse_perm


def closest_psd(a, inv=False):
    """Map a matrix to the closest PSD matrix.

    Args:
        a (tensor): Matrix.
        inv (bool, optional): Also invert `a`.

    Returns:
        tensor: PSD matrix closest to `a` or the inverse of `a`.
    """
    a = B.dense(a)
    a = (a + B.transpose(a)) / 2
    u, s, v = B.svd(a)
    signs = B.matmul(u, v, tr_a=True)
    s = B.maximum(B.diag(signs) * s, 0)
    if inv:
        s = B.where(s == 0, 0, 1 / s)
    return B.mm(u * B.expand_dims(s, axis=-2), v, tr_b=True)


def collect(name="Quantities", **kw_args):
    """Construct a named tuple with certain attributes.

    Args:
        **kw_args (object): Keyword argument specifying the attributes.
        name (str): Name of the named tuple. Defaults to "Quantities".

    Returns:
        :class:`collections.namedtuple`: Named tuple of the model with the
            specified attributes.
    """
    return namedtuple(name, kw_args)(**kw_args)


def autocorr(x, lags=None, cov=False, window=False):
    """Estimate the autocorrelation.

    Args:
        x (vector): Time series to estimate autocorrelation of.
        lags (int, optional): Number of lags. Defaults to all lags.
        cov (bool, optional): Compute covariances rather than correlations. Defaults to
            `False`.
        window (bool, optional): Apply a triangular window to the estimate. Defaults to
            `False`.

    Returns:
        vector: Autocorrelation.
    """
    # Convert to NumPy for compatibility with frameworks.
    x = B.to_numpy(x)

    # Compute autocovariance.
    x = np.reshape(x, -1)  # Flatten the input.
    x = x - np.mean(x)
    k = np.correlate(x, x, mode="full")
    k = k[k.size // 2 :]

    if window:
        # Do not undo the triangular window.
        k = k / x.size
    else:
        # Divide by the precise numbers of estimates.
        k = k / np.arange(x.size, 0, -1)

    # Get the right number of lags.
    if lags is not None:
        k = k[: lags + 1]

    # Divide by estimate of variance if computing correlations.
    if not cov:
        k = k / k[0]

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


def hessian(f, x):
    """Compute the Hessian of a function at a certain input.

    Args:
        f (function): Function to compute Hessian of.
        x (column vector): Input to compute Hessian at.
        differentiable (bool, optional): Make the computation of the Hessian
            differentiable. Defaults to `False`.

    Returns:
        matrix: Hessian.
    """
    if B.rank(x) != 2 or B.shape(x)[1] != 1:
        raise ValueError("Input must be a column vector.")
    # Use RMAD twice to preserve memory.
    hess = jax.jacrev(jax.jacrev(lambda x: f(x[:, None])))(x[:, 0])
    return (hess + B.transpose(hess)) / 2  # Symmetrise to counteract numerical errors.


def maximum_a_posteriori(f, x_init, iters=2000):
    """Compute the MAP estimate.

    Args:
        f (function): Possibly unnormalised log-density.
        x_init (column vector): Starting point to start the optimisation.
        iters (int, optional): Number of optimisation iterations. Defaults to `2000`.

    Returns:
        tensor: MAP estimate.
    """

    def objective(vs_):
        return -f(vs_["x"])

    vs = Vars(B.dtype(x_init))
    vs.unbounded(init=x_init, name="x")
    minimise_l_bfgs_b(objective, vs, iters=iters, jit=True, trace=True)
    return vs["x"]


def laplace_approximation(f, x_init, f_eval=None):
    """Perform a Laplace approximation of a density.

    Args:
        f (function): Possibly unnormalised log-density.
        x_init (column vector): Starting point to start the optimisation.
        f_eval (function): Use this log-density for the evaluation at the MAP estimate.

    Returns:
        tuple[:class:`stheno.Normal`]: Laplace approximation.
    """
    x = maximum_a_posteriori(f, x_init)
    precision = -hessian(f_eval if f_eval is not None else f, x)
    return Normal(x, closest_psd(precision, inv=True))
