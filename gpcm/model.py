from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import lab.jax as B
import wbml.out
from jax.lax import stop_gradient
from matrix import Diagonal
from plum import Dispatcher, Val
from probmods import Model, cast, fit, instancemethod, priormethod
from stheno.jax import Normal
from varz import Vars, minimise_adam, minimise_l_bfgs_b

from .approx import GPCMApproximation, _sample
from .sample import ESS
from .util import closest_psd, estimate_psd, summarise_samples

__all__ = ["AbstractGPCM"]

_dispatch = Dispatcher()


class AbstractGPCM(Model):
    """GPCM model."""

    def __init__(self):
        self.vs = Vars(jnp.float64)

    def __prior__(self):
        # Construct kernel matrices.
        self.K_z = self.compute_K_z()
        self.K_z_inv = B.pd_inv(self.K_z)
        self.K_u = self.compute_K_u()
        self.K_u_inv = B.pd_inv(self.K_u)

        # Construct priors.
        self.p_u = Normal(self.K_u_inv)
        self.p_z = Normal(self.K_z_inv)

        # Construct approximation scheme.
        self.approximation = GPCMApproximation(self)

    def __condition__(self, t, y):
        self.approximation.condition(t, y)

    @instancemethod
    @cast
    def elbo(self, *args, **kw_args):
        return self.approximation.elbo_collapsed_z(*args, **kw_args)

    @instancemethod
    @cast
    def predict(self, *args, **kw_args):
        return self.approximation.predict(*args, **kw_args)

    @instancemethod
    def predict_kernel(self, **kw_args):
        """Predict kernel and normalise prediction.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `50`.

        Returns:
            :class:`collections.namedtuple`: The prediction.
        """
        return summarise_samples(*self.sample_kernel(**kw_args))

    @instancemethod
    def sample_kernel(self, t_k=None, **kw_args):
        """Predict kernel and normalise prediction.

        Args:
            t_k (vector, optional): Inputs to sample kernel at. Will be automatically
                determined if not given.
            num_samples (int, optional): Number of samples to use. Defaults to `50`.

        Returns:
            tuple[vector, tensor]: Tuple containing the inputs of the samples and the
                samples.
        """
        if t_k is None:
            t_k = B.linspace(self.dtype, 0, 1.2 * B.max(self.t_u), 300)

        ks = self.approximation.sample_kernel_samples(t_k, **kw_args)

        # Normalise predicted kernel.
        var_mean = B.mean(ks[:, 0])
        ks = ks / var_mean
        wbml.out.kv("Mean variance of kernel samples", var_mean)

        return t_k, ks

    @instancemethod
    def predict_psd(self, **kw_args):
        """Predict the PSD in dB.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `50`.

        Returns:
            :class:`collections.namedtuple`: Predictions.
        """
        t_k, ks = self.sample_kernel(**kw_args)

        # Estimate PSDs.
        freqs, psds = zip(*[estimate_psd(t_k, k, db=True) for k in ks])
        freqs = freqs[0]
        psds = B.stack(*psds, axis=0)

        return summarise_samples(freqs, psds)

    @instancemethod
    def predict_fourier(self, **kw_args):
        """Predict Fourier features.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `50`.

        Returns:
            tuple: Marginals of the predictions.
        """
        return self.approximation.predict_z(**kw_args)

    @instancemethod
    @cast
    def kernel_approx(self, t1, t2, u):
        """Kernel approximation using inducing variables :math:`u` for the
        impulse response :math:`h`.

        Args:
            t1 (vector): First time input.
            t2 (vector): Second time input.
            u (vector): Values of the inducing variables.

        Returns:
            tensor: Approximation of the kernel matrix broadcasted over `t1` and `t2`.
        """
        # Construct the first part.
        part1 = self.compute_i_hx(t1[:, None], t2[None, :])

        # Construct the second part.
        L_u = B.cholesky(self.K_u)
        inv_L_u = B.trisolve(L_u, B.eye(L_u))
        prod = B.mm(inv_L_u, B.uprank(u, rank=2))
        I_ux = self.compute_I_ux(t1, t2)
        trisolved = B.mm(inv_L_u, I_ux, inv_L_u, tr_c=True)
        part2 = B.trace(trisolved) - B.trace(B.mm(prod, trisolved, prod, tr_a=True))

        return part1 - part2

    @priormethod
    @cast
    def sample(self, t, normalise=False):
        """Sample the kernel then the function.

        Args:
            t (vector): Time points to sample the function at.
            normalise (bool, optional): Normalise the sample of the kernel.
                Defaults to `False`.

        Returns:
            tuple: Tuple containing the kernel matrix and the function.
        """
        u = B.sample(self.compute_K_u())[:, 0]
        K = self.kernel_approx(t, t, u)
        if normalise:
            K = K / K[0, 0]
        f = B.sample(closest_psd(K))[:, 0]
        y = f + B.sqrt(self.noise) * B.randn(f)
        return K, f


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


@fit.dispatch
def fit(model: AbstractGPCM, t, y, method: str = "structured", **kw_args):
    fit(model, t, y, Val(method.lower()), **kw_args)


@fit.dispatch
def fit(model, t, y, method: Val["mean-field-gradient"], iters=1000):
    """Train a model with a mean-field approximation fit using gradient-based
    optimisation.

    Args:
        model (:class:`.gpcm.AbstractGPCM`): Model.
        method (str): Specification of the method. Must be equal to
            :obj:`plum.Val("mean-field-bfgs")`.
        t (vector): Locations of observations.
        y (vector): Observations.
        iters (int, optional): Fixed point iterations. Defaults to `1000`.
    """
    raise NotImplementedError()


@fit.dispatch
def fit(model, t, y, method: Val["mean-field-ca"], iters=1000):
    """Train a model with a mean-field approximation fit using coordinate ascent. This
    approximation does not train hyperparameters.

    Args:
        model (:class:`.gpcm.AbstractGPCM`): Model.
        method (str): Specification of the method. Must be equal to
            :obj:`plum.Val("mean-field-ca")`.
        t (vector): Locations of observations.
        y (vector): Observations.
        iters (int, optional): Fixed point iterations. Defaults to `1000`.
    """
    instance = model()
    ts = instance.approximation.construct_terms(t, y)

    @B.jit
    def compute_q_u(lam, prec):
        lam, prec = instance.approximation.q_u_optimal_mean_field_natural(ts, lam, prec)
        return B.dense(lam), B.dense(prec)

    @B.jit
    def compute_q_z(lam, prec):
        lam, prec = instance.approximation.q_z_optimal_mean_field_natural(ts, lam, prec)
        return B.dense(lam), B.dense(prec)

    def diff(xs, ys):
        total = 0
        for x, y in zip(xs, ys):
            total += B.sqrt(B.mean((x - y) ** 2))
        return total

    # Perform fixed point iterations.
    with wbml.out.Progress(name="Fixed point iterations", total=iters) as progress:
        q_u = tuple(B.dense(x) for x in instance.approximation.q_u)
        q_z = tuple(B.dense(x) for x in instance.approximation.q_z)
        last_q_u = q_u
        last_q_z = q_z

        for _ in range(iters):
            q_u = compute_q_u(*q_z)
            q_z = compute_q_z(*q_u)
            progress({"Difference": diff(q_u + q_z, last_q_u + last_q_z)})
            last_q_u = q_u
            last_q_z = q_z

    # Store result of fixed point iterations.
    instance.approximation.q_u = q_u
    instance.approximation.q_z = q_z


@fit.dispatch
def fit(model, t, y, method: Val["structured"], iters=5000):
    """Train a model using a structured approximation.

    Args:
        model (:class:`.gpcm.AbstractGPCM`): Model.
        method (str): Specification of the method. Must be equal to
            :obj:`plum.Val("structured")`.
        t (vector): Locations of observations.
        y (vector): Observations.
        iters (int, optional): Gibbs sampling iterations. Defaults to `5000`.
    """

    def gibbs_sample(state, iters, u=None, z=None):
        """Perform Gibbs sampling."""
        instance = model()
        ts = instance.approximation.construct_terms(
            B.cast(instance.dtype, t),
            B.cast(instance.dtype, y),
        )

        @B.jit
        def sample_u(state, z):
            q_u = instance.approximation.q_u_optimal_natural(ts, z)
            state, u = _sample(state, q_u, 1)
            return state, B.dense(u)

        @B.jit
        def sample_z(state, u_):
            q_z = instance.approximation.q_z_optimal_natural(ts, u_)
            state, z = _sample(state, q_z, 1)
            return state, B.dense(z)

        # Initialise the state for the Gibbs sampler.
        if u is None:
            state, u = _sample(state, instance.approximation.q_u, 1)
        if z is None:
            state, z = sample_z(state, u)

        # Perform Gibbs sampling.
        with wbml.out.Progress(name="Gibbs sampling", total=iters) as progress:
            us, zs = [], []
            for i in range(iters):
                state, u = sample_u(state, z)
                state, z = sample_z(state, u)
                us.append(u)
                zs.append(z)
                progress()

        return state, us, zs

    # Maintain a random state.
    state = B.create_random_state(model.dtype)

    # Find a good starting point for the optimisation.
    state, us, zs = gibbs_sample(state, iters=500)
    u, z = us[-1], zs[-1]

    def objective(vs_, state, u, z):
        instance = model(vs_)
        ts = instance.approximation.construct_terms(
            B.cast(instance.dtype, t),
            B.cast(instance.dtype, y),
        )
        samples = []
        for _ in range(5):
            q_u = instance.approximation.q_u_optimal_natural(ts, z)
            state, u = _sample(state, q_u, 1)
            q_z = instance.approximation.q_z_optimal_natural(ts, u)
            state, z = _sample(state, q_z, 1)
            u = B.dense(u)
            samples.append(
                instance.approximation.log_Z_u(ts, stop_gradient(u))
                - Normal(instance.K_u).logpdf(B.mm(instance.K_u, stop_gradient(u)))
            )
        return -sum(samples) / len(samples), state, B.dense(u), B.dense(z)

    # Optimise hyperparameters.
    _, state, u, z = minimise_adam(
        objective,
        (model.vs, state, u, z),
        iters=20,
        rate=5e-2,
        trace=True,
        jit=False,
        names=["-q_*"],
    )

    # Fit the mean-field solution for ELBO computation.
    fit(model, t, y, Val("mean-field-ca"))

    # Produce final Gibbs samples and store those samples.
    state, us, zs = gibbs_sample(state, iters, u, z)
    instance = model()
    instance.approximation.q_u_samples = us
    instance.approximation.q_z_samples = zs
