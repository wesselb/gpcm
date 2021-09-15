from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import lab.jax as B
import wbml.out
from matrix import Diagonal
from plum import Val, Dispatcher
from probmods import Model, instancemethod, priormethod, cast, fit
from stheno.jax import Normal
from varz import Vars, minimise_adam, minimise_l_bfgs_b
from .sample import ESS

from .util import summarise_samples, estimate_psd, closest_psd

__all__ = ["AbstractGPCM"]

_dispatch = Dispatcher()


@_dispatch
def _parametrise_q(params, K, K_inv, cov_structure: str):
    n = B.shape(K, 0)
    if cov_structure == "dense":
        cov = params.var.positive_definite(
            K_inv @ (1e-1 * B.eye(n)) @ K_inv,
            shape=(n, n),
        )
    elif cov_structure == "posterior":
        diag = Diagonal(params.var.positive(1e-1, shape=(n,)))
        cov = K_inv - B.pd_inv(K + diag)
    else:
        raise ValueError('`cov_structure` must be "dense" or "posterior".')
    return Normal(params.mean.unbounded(shape=(n, 1)), cov)


def _m1s_m2s_to_marginals(m1s, m2s):
    m1 = B.mean(B.stack(*m1s, axis=0), axis=0)
    m2 = B.mean(B.stack(*m2s, axis=0), axis=0)
    return m1, m2 - m1 ** 2


class GPCMApproximation:
    """Approximation for the GPCM.

    Args:
        model (:class:`probmods.Model`): Instantiated model.
        cov_structure (str, optional): Covariance structure. Must be "dense" or
            "posterior". Defaults to "posterior".
    """

    @_dispatch
    def __init__(self, model: Model, cov_structure: str = "posterior"):
        self.model = model
        self.cov_structure = cov_structure

        # Construct initial variational approximations.
        self._q_i = -1
        self.p_u = self.model.p_u
        self.p_z = self.model.p_z
        self._new_qs()

    def _new_qs(self):
        self._q_i += 1
        self.q_u = _parametrise_q(
            self.model.ps.q_u[self._q_i],
            self.model.K_u,
            self.model.K_u_inv,
            self.cov_structure,
        )
        self.q_z = _parametrise_q(
            self.model.ps.q_z[self._q_i],
            self.model.K_z,
            self.model.K_z_inv,
            self.cov_structure,
        )

    @property
    def p_u_optimal_samples(self):
        """list[tensor]: After conditioning, get samples from the previous optimal
        :math:`q(u)`, which is now the optimal prior."""
        samples = self.model.ps.q_u[self._q_i - 1].samples()
        return B.unstack(samples, axis=1, squeeze=False)

    @property
    def q_u_optimal_samples(self):
        """list[tensor]: Samples from the optimal :math:`q(u)`."""
        samples = self.model.ps.q_u[self._q_i].samples()
        return B.unstack(samples, axis=1, squeeze=False)

    @q_u_optimal_samples.setter
    @_dispatch
    def q_u_optimal_samples(self, samples: list):
        # We need a setter, because these won't be trainable through gradients.
        self.model.ps.q_u[self._q_i].samples.delete()
        samples = B.stack(*[B.flatten(x) for x in samples], axis=1)
        self.model.ps.q_u[self._q_i].samples.unbounded(init=samples, trainable=False)

    @property
    def p_z_optimal_samples(self):
        """list[tensor]: After conditioning, get samples from the previous optimal
        :math:`q(z)`, which is now the optimal prior."""
        samples = self.model.ps.q_z[self._q_i - 1].samples()
        return B.unstack(samples, axis=1, squeeze=False)

    @property
    def q_z_optimal_samples(self):
        """list[tensor]: Samples from the optimal :math:`q(z)`."""
        samples = self.model.ps.q_z[self._q_i].samples()
        return B.unstack(samples, axis=1, squeeze=False)

    @q_z_optimal_samples.setter
    @_dispatch
    def q_z_optimal_samples(self, samples: list):
        # We need a setter, because these won't be trainable through gradients.
        self.model.ps.q_z[self._q_i].samples.delete()
        samples = B.stack(*[B.flatten(x) for x in samples], axis=1)
        self.model.ps.q_z[self._q_i].samples.unbounded(init=samples, trainable=False)

    def condition(self, t, y):
        """Make the current variational approximation the prior and create new
        variational approximations for a new posterior.

        Args:
            t (vector): Locations of observations.
            y (vector): Observations.
        """
        self.ts = self.construct_terms(t, y)
        self.p_u = self.q_u
        self.p_z = self.q_z
        self._new_qs()

    def construct_terms(self, t, y=None):
        """Construct quantities required.

        Args:
            t (vector): Locations of observations.
            y (vector, optional): Observations.
        """
        ts = SimpleNamespace()

        ts.n = B.length(t)

        # Construct integrals.
        ts.I_hx = self.model.compute_i_hx(t, t)
        ts.I_ux = self.model.compute_I_ux()
        ts.I_hz = self.model.compute_I_hz(t)
        ts.I_uz = self.model.compute_I_uz(t)

        # Do some precomputations.
        ts.I_hx_sum = B.sum(ts.I_hx, axis=0)
        ts.I_hz_sum = B.sum(ts.I_hz, axis=0)
        ts.I_ux_sum = ts.n * ts.I_ux

        ts.K_u_squeezed = B.mm(ts.I_uz, self.model.K_u_inv, ts.I_uz, tr_a=True)
        ts.K_z_squeezed = B.mm(ts.I_uz, self.model.K_z_inv, ts.I_uz, tr_c=True)
        ts.A_sum = ts.I_ux_sum - B.sum(ts.K_z_squeezed, axis=0)
        ts.B_sum = ts.I_hz_sum - B.sum(ts.K_u_squeezed, axis=0)
        ts.c_sum = (
            ts.I_hx_sum
            - B.sum(self.model.K_u_inv * ts.I_ux_sum)
            - B.sum(self.model.K_z_inv * ts.I_hz_sum)
            + B.sum(self.model.K_u_inv * ts.K_z_squeezed)
        )

        if y is not None:
            ts.y = y
            # Weight by data.
            ts.I_uz_sum = B.sum(y[:, None, None] * ts.I_uz, axis=0)

        return ts

    @_dispatch
    def q_z_optimal(self, ts, u: B.Numeric):
        """Compute the optimal :math:`q(z|u)`.

        Args:
            ts (object): Terms.
            u (tensor): Sample for :math:`u`.

        Returns:
            :class:`stheno.Normal`: Optimal :math:`q(z|u)`.
        """
        part = B.mm(ts.I_uz, u, tr_a=True)
        inv_cov_z = self.model.K_z + 1 / self.model.noise * (
            ts.B_sum + B.sum(B.mm(part, part, tr_b=True), axis=0)
        )
        inv_cov_z_mu_z = 1 / self.model.noise * B.mm(ts.I_uz_sum, u, tr_a=True)
        cov_z = B.pd_inv(inv_cov_z)
        return Normal(B.mm(cov_z, inv_cov_z_mu_z), cov_z)

    @_dispatch
    def q_z_optimal_mean_field(self, ts, q_u: Normal):
        """Compute the optimal :math:`q(z)` given some :math:`q(u)` in the mean field
        approximation.

        Args:
            ts (object): Terms.
            u (:class:`stheno.Normal`): Current :math:`q(u)`.

        Returns:
            :class:`stheno.Normal`: Optimal :math:`q(z)`.
        """
        inv_cov_z = self.model.K_z + 1 / self.model.noise * (
            ts.B_sum + B.sum(B.mm(ts.I_uz, q_u.m2, ts.I_uz, tr_a=True), axis=0)
        )
        inv_cov_z_mu_z = 1 / self.model.noise * B.mm(ts.I_uz_sum, q_u.mean, tr_a=True)
        cov_z = B.pd_inv(inv_cov_z)
        return Normal(B.mm(cov_z, inv_cov_z_mu_z), cov_z)

    @_dispatch
    def q_u_optimal(self, ts, z: B.Numeric):
        """Compute the optimal :math:`q(u|z)`.

        Args:
            ts (object): Terms.
            u (tensor): Sample for :math:`z`.

        Returns:
            :class:`stheno.Normal`: Optimal :math:`q(u|z)`.
        """
        part = B.mm(ts.I_uz, z)
        inv_cov_u = self.model.K_u + 1 / self.model.noise * (
            ts.A_sum + B.sum(B.mm(part, part, tr_b=True), axis=0)
        )
        inv_cov_u_mu_u = 1 / self.model.noise * B.mm(ts.I_uz_sum, z)
        cov_u = B.pd_inv(inv_cov_u)
        return Normal(B.mm(cov_u, inv_cov_u_mu_u), cov_u)

    @_dispatch
    def q_u_optimal_mean_field(self, ts, q_z: Normal):
        """Compute the optimal :math:`q(u)` given some :math:`q(z)` in the mean field
        approximation.

        Args:
            ts (object): Terms.
            u (:class:`stheno.Normal`): Current :math:`q(z)`.

        Returns:
            :class:`stheno.Normal`: Optimal :math:`q(u)`.
        """
        inv_cov_u = self.model.K_u + 1 / self.model.noise * (
            ts.A_sum + B.sum(B.mm(ts.I_uz, q_z.m2, ts.I_uz, tr_c=True), axis=0)
        )
        inv_cov_u_mu_u = 1 / self.model.noise * B.mm(ts.I_uz_sum, q_z.mean)
        cov_u = B.pd_inv(inv_cov_u)
        return Normal(B.mm(cov_u, inv_cov_u_mu_u), cov_u)

    def log_Z_u(self, ts, u):
        """Compute the normalising constant term of the optimal :math:`q(u)`.

        Args:
            ts (object): Terms.
            u (tensor): Sample for :math:`u` to use.

        Returns:
            scalar: Normalising constant term.
        """
        q_z = self.q_z_optimal(ts, u)
        return (
            -0.5 * ts.n * B.log(2 * B.pi * self.model.noise)
            + (
                (-0.5 / self.model.noise)
                * (B.sum(ts.y ** 2) + B.sum(u * B.mm(ts.A_sum, u)) + ts.c_sum)
            )
            + -0.5 * B.logdet(self.model.K_z)
            + 0.5 * B.logdet(q_z.var)
            + 0.5 * B.iqf(q_z.var, q_z.mean)[0, 0]
        )

    def log_Z_z(self, ts, z):
        """Compute the normalising constant term of the optimal :math:`q(z)`.

        Args:
            ts (object): Terms.
            z (tensor): Sample for :math:`z` to use.

        Returns:
            scalar: Normalising constant term.
        """
        q_u = self.q_u_optimal(ts, z)
        return (
            -0.5 * ts.n * B.log(2 * B.pi * self.model.noise)
            + (
                (-0.5 / self.model.noise)
                * (B.sum(ts.y ** 2) + B.sum(z * B.mm(ts.B_sum, z)) + ts.c_sum)
            )
            + -0.5 * B.logdet(self.model.K_u)
            + 0.5 * B.logdet(q_u.var)
            + 0.5 * B.iqf(q_u.var, q_u.mean)[0, 0]
        )

    @_dispatch
    def elbo_collapsed_z(self, state, t, y, *, num_samples=5):
        """Compute an estimate of the ELBO collapsed over :math:`q(z|u)`.

        Args:
            t (vector): Locations of observations.
            y (vector): Observations.
            num_samples (int, optional): Number of samples to use. Defaults to `5`.

        Returns:
            scalar: ELBO.
        """
        ts = self.construct_terms(t, y)
        state, samples = self.q_u.sample(state, num_samples)
        rec_samples = [
            self.log_Z_u(ts, u) for u in B.unstack(samples, axis=1, squeeze=False)
        ]
        return state, sum(rec_samples) / num_samples - self.q_u.kl(self.p_u)

    @_dispatch
    def elbo_collapsed_z(self, t, y, *, num_samples=5):
        state = B.global_random_state(self.dtype)
        state, elbo = self.elbo_collapsed_z(state, t, y, num_samples=num_samples)
        B.set_global_random_state(state)
        return elbo

    @_dispatch
    def elbo_collapsed_u(self, state, t, y, *, num_samples=5):
        """Compute an estimate of the ELBO collapsed over :math:`q(u|z)`.

        Args:
            t (vector): Locations of observations.
            y (vector): Observations.
            num_samples (int, optional): Number of samples to use. Defaults to `5`.

        Returns:
            scalar: ELBO.
        """
        ts = self.construct_terms(t, y)
        state, samples = self.q_z.sample(state, num_samples)
        rec_samples = [
            self.log_Z_z(ts, z) for z in B.unstack(samples, axis=1, squeeze=False)
        ]
        return state, sum(rec_samples) / num_samples - self.q_z.kl(self.p_z)

    @_dispatch
    def elbo_collapsed_u(self, t, y, *, num_samples=5):
        state = B.global_random_state(self.dtype)
        state, elbo = self.elbo_collapsed_u(state, t, y, num_samples=num_samples)
        B.set_global_random_state(state)
        return elbo

    def elbo_mean_field(self, t, y):
        pass

    def predict_samples(self, t, num_samples=50):
        """Predict.

        Args:
            t (vector): Points to predict at.
            num_samples (int, optional): Number of samples to use. Defaults to `50`.

        Returns:
            tuple: Tuple containing the mean and standard deviation of the
                predictions.
        """
        ts = self.construct_terms(t)
        m1s, m2s = zip(
            *[
                self._predict_moments(ts, u, B.outer(u), z, B.outer(z))
                for u, z in zip(
                    self.p_u_optimal_samples[:num_samples],
                    self.p_z_optimal_samples[:num_samples],
                )
            ]
        )
        return _m1s_m2s_to_marginals(m1s, m2s)

    def predict_collapsed_z(self, t, num_samples=50):
        """Predict.

        Args:
            t (vector): Points to predict at.
            num_samples (int, optional): Number of samples to use. Defaults to `50`.

        Returns:
            tuple: Tuple containing the mean and standard deviation of the
                predictions.
        """
        ts = self.construct_terms(t)
        m1s, m2s = [], []
        for u in B.unstack(self.p_u.sample(num_samples), axis=1, squeeze=False):
            q_z = self.q_z_optimal(self.ts, u)
            m1, m2 = self._predict_moments(ts, u, B.outer(u), q_z.mean, q_z.m2)
            m1s.append(m1)
            m2s.append(m2)
        return _m1s_m2s_to_marginals(m1s, m2s)

    def predict_mean_field(self, t):
        """Predict.

        Args:
            t (vector): Points to predict at.

        Returns:
            tuple: Tuple containing the mean and standard deviation of the
                predictions.
        """
        m1, m2 = self._predict_moments(
            self.construct_terms(t),
            self.p_u.mean,
            self.p_u.var,
            self.p_z.mean,
            self.p_z.var,
        )
        return m1, m2 - m1 ** 2

    def _predict_moments(self, ts, u, u2, z, z2):
        # Compute first moment.
        m1 = B.flatten(B.mm(u, ts.I_uz, z, tr_a=True))

        # Compute second moment.
        A = ts.I_ux - ts.K_z_squeezed
        B_ = ts.I_hz - ts.K_u_squeezed
        c = (
            ts.I_hx
            - B.sum(self.model.K_u_inv * ts.I_ux)
            - B.sum(B.dense(self.model.K_z_inv) * ts.I_hz, axis=(1, 2))
            + B.sum(B.dense(self.model.K_u_inv) * ts.K_z_squeezed, axis=(1, 2))
        )
        m2 = (
            B.sum(A * u2, axis=(1, 2))
            + B.sum(B_ * z2, axis=(1, 2))
            + c
            + B.sum(u2 * B.mm(ts.I_uz, z2, ts.I_uz, tr_c=True), axis=(1, 2))
        )

        return m1, m2

    def sample_kernel_samples(self, t_k, num_samples=200):
        """Sample kernel.

        Args:
            t_k (vector): Time point to sample at.
            num_samples (int, optional): Number of samples to use. Defaults to `200`.

        Returns:
            tensor: Samples.
        """
        return B.stack(
            *[
                self._sample_kernel(t_k, u)
                for u in self.p_u_optimal_samples[:num_samples]
            ],
            axis=0
        )

    def sample_kernel_collapsed_u(self, t_k, num_samples=200):
        """Sample kernel.

        Args:
            t_k (vector): Time point to sample at.
            num_samples (int, optional): Number of samples to use. Defaults to `200`.

        Returns:
            tensor: Samples.
        """
        samples = self.p_u.sample(num_samples)
        return B.stack(
            *[
                self._sample_kernel(t_k, u)
                for u in B.unstack(samples, axis=1, squeeze=False)
            ],
            axis=0
        )

    def _sample_kernel(self, t_k, u):
        return B.flatten(
            self.model.kernel_approx(
                t_k,
                B.zero(u)[None],
                B.flatten(B.dense(B.matmul(self.model.K_u, u))),
            )
        )

    def predict_z_samples(self, num_samples=200):
        """Predict Fourier features.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `200`.

        Returns:
            tuple: Marginals of the predictions.
        """
        samples = [B.flatten(x) for x in self.p_z_optimal_samples[:num_samples]]
        return _m1s_m2s_to_marginals(samples, [x ** 2 for x in samples])

    def predict_z_collapsed_z(self, num_samples=200):
        """Predict Fourier features.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `200`.

        Returns:
            tuple: Marginals of the predictions.
        """
        m1s = []
        m2s = []
        for u in B.unstack(self.p_u.sample(num_samples), axis=1, squeeze=False):
            p_z = self.q_z_optimal(self.ts, u)
            m1s.append(B.flatten(p_z.mean))
            m2s.append(B.diag(p_z.var) + m1s[-1] ** 2)
        return _m1s_m2s_to_marginals(m1s, m2s)

    def predict_z_mean_field(self):
        """Predict Fourier features.

        Returns:
            tuple: Marginals of the predictions.
        """
        return self.p_z.marginals()


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
        return self.approximation.elbo_collapsed_u(*args, **kw_args)

    @instancemethod
    @cast
    def predict(self, *args, **kw_args):
        return self.approximation.predict_samples(*args, **kw_args)

    @instancemethod
    def predict_kernel(self, num_samples=200):
        """Predict kernel and normalise prediction.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `200`.

        Returns:
            :class:`collections.namedtuple`: The prediction.
        """
        return summarise_samples(*self.sample_kernel(num_samples=num_samples))

    @instancemethod
    def sample_kernel(self, t_k=None, num_samples=200):
        """Predict kernel and normalise prediction.

        Args:
            t_k (vector, optional): Inputs to sample kernel at. Will be automatically
                determined if not given.
            num_samples (int, optional): Number of samples to use. Defaults to `200`.

        Returns:
            tuple[vector, tensor]: Tuple containing the inputs of the samples and the
                samples.
        """
        if t_k is None:
            t_k = B.linspace(self.dtype, 0, 1.2 * B.max(self.t_u), 300)

        ks = self.approximation.sample_kernel_samples(t_k, num_samples=num_samples)

        # Normalise predicted kernel.
        var_mean = B.mean(ks[:, 0])
        ks = ks / var_mean
        wbml.out.kv("Mean variance of kernel samples", var_mean)

        return t_k, ks

    @instancemethod
    def predict_psd(self, num_samples=200):
        """Predict the PSD in dB.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `200`.

        Returns:
            :class:`collections.namedtuple`: Predictions.
        """
        t_k, ks = self.sample_kernel(num_samples=num_samples)

        # Estimate PSDs.
        freqs, psds = zip(*[estimate_psd(t_k, k, db=True) for k in ks])
        freqs = freqs[0]
        psds = B.stack(*psds, axis=0)

        return summarise_samples(freqs, psds)

    @instancemethod
    def predict_fourier(self, num_samples=200):
        """Predict Fourier features.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `200`.

        Returns:
            tuple: Marginals of the predictions.
        """
        return self.approximation.predict_z_samples(num_samples=num_samples)

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
        inv_L_u = B.dense(B.trisolve(L_u, B.eye(L_u)))
        prod = B.mm(inv_L_u, B.uprank(u, rank=2))
        I_ux = self.compute_I_ux(t1, t2)
        trisolved = B.mm(inv_L_u, I_ux, inv_L_u, tr_c=True)
        # Black butchers the following lines, so we format it manually.
        # fmt: off
        part2 = (
            B.trace(trisolved, axis1=2, axis2=3)
            - B.trace(B.mm(prod, trisolved, prod, tr_a=True), axis1=2, axis2=3)
        )
        # fmt: on

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
def fit(model: AbstractGPCM, t, y, method: str = "laplace", **kw_args):
    fit(model, t, y, Val(method.lower()), **kw_args)


@fit.dispatch
def fit(model, t, y, method: Val["laplace"]):
    """Train using a Laplace approximation.

    Args:
        method (str): Specification of the method. Must be equal to
            :obj:`plum.Val("laplace")`.
        t (vector): Locations of observations.
        y (vector): Observations.
    """
    # Initialise with a Laplace approximation.
    instance = model()
    q_u = laplace_approximation(
        partial(instance.logpdf_optimal_q_u, t, y),
        B.randn(instance.dtype, instance.n_u, 1),
    )
    model.vs.struct.q_u[-1].mean.assign(q_u.mean)
    model.vs.struct.q_u[-1].var.assign(q_u.var)


@fit.dispatch
def fit(
    model: AbstractGPCM, t, y, method: Val["laplace-vi"], iters=100, fix_noise=False
):
    """Train a model by optimising a Laplace approximation within a variational
    lower bound.

    Args:
        model (:class:`.gpcm.AbstractGPCM`): Model.
        method (str): Specification of the method. Must be equal to
            :obj:`plum.Val("laplace-vi")`.
        t (vector): Locations of observations.
        y (vector): Observations.
        iters (int, optional): Enclosing VI training iterations. Defaults to `0`.
        fix_noise (bool, optional): Fix the noise during training. Defaults to `False`.
    """
    # Persistent state during optimisation.
    state = {"u": B.randn(model.dtype, model.n_u, 1)}

    def objective(vs_, random_state):
        instance = model(vs_)

        # Create an instance without gradients for the Laplace approximation
        instance_detached = model(vs_.copy(f=jax.lax.stop_gradient))

        # Perform Laplace approximation.
        dist, state["u"] = laplace_approximation(
            partial(instance_detached.logpdf_optimal_q_u, t, y),
            state["u"],
            f_eval=partial(instance.logpdf_optimal_q_u, t, y),
        )

        # Use a high number of samples for high-quality gradients.
        random_state, elbo = instance.elbo(
            random_state,
            B.cast(vs_.dtype, t),  # Prevent conversion back to NumPy.
            y,
            num_samples=50,
        )
        return -elbo, random_state

    # Determine the names of the variables to optimise.
    model()  # Instantiate the model to ensure that all variables exist.
    names = model.vs.names
    if fix_noise:
        names = list(set(names) - {"noise"})

    # Perform optimisation.
    random_state = B.create_random_state(model.dtype)
    minimise_adam(
        objective,
        (model.vs, random_state),
        iters=iters,
        trace=True,
        rate=5e-2,
        names=names,
    )

    # Create final Laplace approximation.
    fit(model, t, y, method="laplace")


@fit.dispatch
def fit(model, t, y, method: Val["vi"], iters=1000, fix_noise=False):
    """Train a model using VI.

    Args:
        model (:class:`.gpcm.AbstractGPCM`): Model.
        method (str): Specification of the method. Must be equal to
            :obj:`plum.Val("vi")`.
        t (vector): Locations of observations.
        y (vector): Observations.
        iters (int, optional): Training iterations. Defaults to `1000`.
        fix_noise (bool, optional): Fix the noise during training. Defaults to `False`.
    """

    def objective(vs_, state):
        state, elbo = model(vs_).elbo(
            state,
            B.cast(vs_.dtype, t),  # Prevent conversion to NumPy.
            y,
            num_samples=5,
        )
        return -elbo, state

    # Determine the names of the variables to optimise.
    model()  # Instantiate the model to ensure that all variables exist.
    names = model.vs.names
    if fix_noise:
        names = list(set(names) - {"noise"})

    # Perform optimisation.
    state = B.create_random_state(model.dtype)
    # Initialise the variational parameters. The learning rate can be fairly high here.
    minimise_adam(
        objective,
        (model.vs, state),
        iters=10,
        trace=True,
        rate=5e-2,
        names="q_*",
        jit=True,
    )
    # Use the other half the iterations to jointly optimise everything. The learning
    # rate needs to be lower here.
    minimise_adam(
        objective,
        (model.vs, state),
        iters=iters // 2,
        trace=True,
        rate=2e-2,
        names=names,
        jit=True,
    )


@fit.dispatch
def fit(model, t, y, method: Val["ess"], iters=1000):
    """Train a model using VI.

    Args:
        model (:class:`.gpcm.AbstractGPCM`): Model.
        method (str): Specification of the method. Must be equal to
            :obj:`plum.Val("vi")`.
        t (vector): Locations of observations.
        y (vector): Observations.
        iters (int, optional): Training iterations. Defaults to `1000`.
    """
    instance = model()
    x_map = maximum_a_posteriori(
        B.jit(partial(instance.logpdf_optimal_q_u, t, y)),
        B.randn(instance.dtype, instance.n_u, 1),
        iters=50,
    )
    sampler = ESS(
        B.jit(partial(instance.logpdf_optimal_q_u, t, y, prior=False)),
        instance.p_u.sample,
        x_map,
    )
    model.samples = sampler.sample(num=500, trace=True)[-400:]


@fit.dispatch
def fit(model, t, y, method: Val["gibbs"], iters=500, num_samples=50):
    """Train a model using Gibbs sampling.

    Args:
        model (:class:`.gpcm.AbstractGPCM`): Model.
        method (str): Specification of the method. Must be equal to
            :obj:`plum.Val("gibbs")`.
        t (vector): Locations of observations.
        y (vector): Observations.
        iters (int, optional): Gibbs sampling iterations. Defaults to `100`.
        num_samples (int, optinal): Number of samples.
    """
    instance = model()
    ts = instance.approximation.construct_terms(t, y)

    @B.jit
    def sample_u(state, z):
        q_u = instance.approximation.q_u_optimal(ts, z)
        return q_u.sample(state)

    @B.jit
    def sample_z(state, u):
        q_z = instance.approximation.q_z_optimal(ts, u)
        return q_z.sample(state)

    # Perform Gibbs sampling.
    state = B.create_random_state(model.dtype)
    us, zs = [], []
    with wbml.out.Progress(name="Gibbs sampling", total=num_samples) as progress:
        for _ in range(num_samples):
            state, u = instance.p_u.sample(state)
            state, z = sample_z(state, u)
            for _ in range(iters):
                state, u = sample_u(state, z)
                state, z = sample_z(state, u)
            us.append(u)
            zs.append(z)
            progress()

    # Store samples.
    instance.approximation.q_u_optimal_samples = us
    instance.approximation.q_z_optimal_samples = zs
