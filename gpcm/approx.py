from types import SimpleNamespace

import lab.jax as B
import wbml.out
from jax.lax import stop_gradient
from plum import Dispatcher, Union
from probmods import Model, fit
from varz import minimise_l_bfgs_b, minimise_adam
from varz.spec import Struct

from .normal import NaturalNormal

__all__ = ["Structured", "MeanField"]

_dispatch = Dispatcher()


@_dispatch
def _parametrise_natural(params: Struct, n: B.Int):
    return NaturalNormal(
        params.lam.unbounded(B.randn(n, 1), shape=(n, 1)),
        params.prec.positive_definite(1e-1 * B.eye(n), shape=(n, n)),
    )


@_dispatch
def _columns(xs: B.Numeric, num: Union[None, B.Int] = None):
    if num is None:
        num = B.shape(xs, 1)
    return [xs[:, i : i + 1] for i in range(min(B.shape(xs, 1), num))]


class Approximation:
    """Approximation of the GPCM.

    Args:
        model (:class:`probmods.Model`): Instantiated GPCM.
    """

    @_dispatch
    def __init__(self, model: Model):
        self.model = model
        self._q_i = 0  # Count at which variational posterior we are.

    @property
    def p_u(self):
        """:class:`stheno.Normal`: Current prior for :math:`u`."""
        if self._q_i == 0:
            return NaturalNormal(0, self.model.K_u)
        else:
            return _parametrise_natural(
                self.model.ps.q_u[self._q_i - 1],
                self.model.n_u,
            )

    @property
    def q_u(self):
        """:class:`stheno.Normal`: Current variational posterior for :math:`u`."""
        return _parametrise_natural(self.model.ps.q_u[self._q_i], self.model.n_u)

    @q_u.setter
    @_dispatch
    def q_u(self, dist: NaturalNormal):
        self.q_u  # Ensure that it is initialised.
        self.model.ps.q_u[self._q_i].lam.assign(B.dense(dist.lam))
        self.model.ps.q_u[self._q_i].prec.assign(B.dense(dist.prec))

    @property
    def p_z(self):
        """:class:`stheno.Normal`: Current prior for :math:`z`."""
        if self._q_i == 0:
            return NaturalNormal(0, self.model.K_z)
        else:
            return _parametrise_natural(
                self.model.ps.q_z[self._q_i - 1],
                self.model.n_z,
            )

    @property
    def q_z(self):
        """:class:`stheno.Normal`: Variational posterior for :math:`z`."""
        return _parametrise_natural(self.model.ps.q_z[self._q_i], self.model.n_z)

    @q_z.setter
    @_dispatch
    def q_z(self, dist: NaturalNormal):
        self.q_z  # Ensure that it is initialised.
        self.model.ps.q_z[self._q_i].lam.assign(B.dense(dist.lam))
        self.model.ps.q_z[self._q_i].prec.assign(B.dense(dist.prec))

    def ignore_qs(self, previous, current):
        """Get a list of regexes that ignore variables corresponding to approximate
        posteriors.

        Args:
            previous (bool): Ignore all previous approximate posteriors.
            current (bool or str): Current approximate posteriors to ignore. Set to
                `True` or `False` to ignore all or none or to `z` or `u` to ignore
                a specific approximate posterior.

        Returns:
            list[str]: Appropriate list of regexes.
        """
        names = []
        if previous:
            for i in range(self._q_i):
                names.append("-" + self.model.ps.q_u[i].all())
                names.append("-" + self.model.ps.q_z[i].all())
        if current is True:
            names.append("-" + self.model.ps.q_u[self._q_i].all())
            names.append("-" + self.model.ps.q_z[self._q_i].all())
        elif current == "u":
            names.append("-" + self.model.ps.q_u[self._q_i].all())
        elif current == "z":
            names.append("-" + self.model.ps.q_z[self._q_i].all())
        else:
            raise ValueError(f'Invalid value "{current}" for `current`.')
        return names

    def condition(self, t, y):
        """Make the current variational approximation the prior and create a new
        variational posterior for further observations.

        Args:
            t (vector): Locations of observations.
            y (vector): Observations.
        """
        self._q_i += 1
        self.ts = self.construct_terms(t, y)

    def construct_terms(self, t, y=None):
        """Construct commonly required quantities.

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
            # It would be more efficient to first `B.sum(ts.K_z_squeezed, axis=0)`, but
            # for some reason that results in a segmentation fault when run on with the
            # JIT on the GPU. I'm not sure what's going on...
            + B.sum(self.model.K_u_inv * ts.K_z_squeezed)
        )

        if y is not None:
            ts.y = y
            ts.I_uz_sum = B.sum(y[:, None, None] * ts.I_uz, axis=0)  # Weight by data.

        return ts

    @_dispatch
    def q_u_optimal(self, ts: SimpleNamespace, z: B.Numeric):
        """Compute the optimal :math:`q(u|z)`.

        Args:
            ts (:class:`types.SimpleNamespace`): Terms.
            u (tensor): Sample for :math:`z`.

        Returns:
            tuple[tensor, tensor]: Natural parameters.
        """
        return self._q_optimal(
            ts.A_sum,
            ts.I_uz,
            ts.I_uz_sum,
            self.model.K_u,
            z,
        )

    @_dispatch
    def q_u_optimal_mean_field(self, ts: SimpleNamespace, q_z: NaturalNormal):
        """Compute the optimal :math:`q(u)` in the mean-field approximation.

        Args:
            ts (:class:`types.SimpleNamespace`): Terms.
            q_z (:class:`.normal.NaturalNormal`): Current estimate for :math:`q(z)`.

        Returns:
            tuple[tensor, tensor]: Natural parameters.
        """
        return self._q_optimal(
            ts.A_sum,
            ts.I_uz,
            ts.I_uz_sum,
            self.model.K_u,
            q_z.mean,
            q_z.m2,
        )

    @_dispatch
    def q_z_optimal(self, ts: SimpleNamespace, u: B.Numeric):
        """Compute the optimal :math:`q(z|u)`.

        Args:
            ts (:class:`types.SimpleNamespace`): Terms.
            u (tensor): Sample for :math:`u`.

        Returns:
            tuple[tensor, tensor]: Natural parameters.
        """
        return self._q_optimal(
            ts.B_sum,
            B.transpose(ts.I_uz),
            B.transpose(ts.I_uz_sum),
            self.model.K_z,
            u,
        )

    @_dispatch
    def q_z_optimal_mean_field(self, ts: SimpleNamespace, q_u: NaturalNormal):
        """Compute the optimal :math:`q(z)` in the mean-field approximation.

        Args:
            ts (:class:`types.SimpleNamespace`): Terms.
            q_u (:class:`.normal.NaturalNormal`): Current estimate for :math:`q(u)`.

        Returns:
            tuple[tensor, tensor]: Natural parameters.
        """
        return self._q_optimal(
            ts.B_sum,
            B.transpose(ts.I_uz),
            B.transpose(ts.I_uz_sum),
            self.model.K_z,
            q_u.mean,
            q_u.m2,
        )

    def _q_optimal(self, A, I, I_sum, K, x, x2=None):
        if x2 is not None:
            inner = B.mm(I, x2, I, tr_c=True)
        else:
            # This is _much_ more efficient!
            part = B.mm(I, x)
            inner = B.mm(part, part, tr_b=True)
        return NaturalNormal(
            B.mm(I_sum, x) / self.model.noise,
            K + (A + B.sum(inner, axis=0)) / self.model.noise,
        )

    def _predict_moments(self, ts, u, u2, z, z2):
        # Compute first moment.
        m1 = B.flatten(B.mm(u, ts.I_uz, z, tr_a=True))

        # Compute second moment.
        A = ts.I_ux - ts.K_z_squeezed
        B_ = ts.I_hz - ts.K_u_squeezed
        c = (
            ts.I_hx
            - B.sum(self.model.K_u_inv * ts.I_ux)
            - B.sum(self.model.K_z_inv * ts.I_hz, axis=(1, 2))
            + B.sum(self.model.K_u_inv * ts.K_z_squeezed, axis=(1, 2))
        )
        m2 = (
            B.sum(A * u2, axis=(1, 2))
            + B.sum(B_ * z2, axis=(1, 2))
            + c
            + B.sum(u2 * B.mm(ts.I_uz, z2, ts.I_uz, tr_c=True), axis=(1, 2))
        )

        return m1, m2

    def _sample_kernel(self, t_k, u):
        return B.flatten(
            self.model.kernel_approx(
                t_k,
                B.zero(u)[None],
                B.flatten(B.matmul(self.model.K_u, u)),
            )
        )


@_dispatch
def _fit_mean_field_ca(instance: Model, t, y, iters: B.Int = 1000):
    """Train an instance with a mean-field approximation using coordinate ascent.

    Args:
        instance (:class:`.gpcm.AbstractGPCM`): Instantiated model.
        t (vector): Locations of observations.
        y (vector): Observations.
        iters (int, optional): Fixed point iterations. Defaults to `1000`.
    """
    ts = instance.approximation.construct_terms(t, y)

    @B.jit
    def compute_q_u(lam, prec):
        q_z = NaturalNormal(lam, prec)
        q_u = instance.approximation.q_u_optimal_mean_field(ts, q_z)
        return B.dense(q_u.lam), B.dense(q_u.prec)

    @B.jit
    def compute_q_z(lam, prec):
        q_u = NaturalNormal(lam, prec)
        q_z = instance.approximation.q_z_optimal_mean_field(ts, q_u)
        return B.dense(q_z.lam), B.dense(q_z.prec)

    def diff(xs, ys):
        total = 0
        for x, y in zip(xs, ys):
            total += B.sqrt(B.mean((x - y) ** 2))
        return total

    # Perform fixed point iterations.
    with wbml.out.Progress(name="Fixed point iterations", total=iters) as progress:
        q_u = instance.approximation.q_u
        q_z = instance.approximation.q_z
        # To be able to use the JIT, we must pass around plain tensors.
        q_u = (B.dense(q_u.lam), B.dense(q_u.prec))
        q_z = (B.dense(q_z.lam), B.dense(q_z.prec))
        last_q_u = q_u
        last_q_z = q_z

        for _ in range(iters):
            q_u = compute_q_u(*q_z)
            q_z = compute_q_z(*q_u)
            progress({"Difference": diff(q_u + q_z, last_q_u + last_q_z)})
            last_q_u = q_u
            last_q_z = q_z

    # Store result of fixed point iterations.
    instance.approximation.q_u = NaturalNormal(*q_u)
    instance.approximation.q_z = NaturalNormal(*q_z)


class Structured(Approximation):
    """Structured approximation of the GPCM."""

    @property
    def p_u_samples(self):
        """tensor: Samples for the current prior for :math:`u`."""
        return self.model.ps.q_u[self._q_i - 1].samples()

    @property
    def q_u_samples(self):
        """tensor: Samples for the current variational posterior for :math:`u`."""
        return self.model.ps.q_u[self._q_i].samples()

    @q_u_samples.setter
    @_dispatch
    def q_u_samples(self, samples: list):
        # We need a setter, because these won't be trainable through gradients.
        self.model.ps.q_u[self._q_i].samples.delete()
        samples = B.stack(*[B.flatten(x) for x in samples], axis=1)
        self.model.ps.q_u[self._q_i].samples.unbounded(init=samples, visible=False)

    @property
    def p_z_samples(self):
        """tensor: Samples from the current prior for :math:`z`."""
        return self.model.ps.q_z[self._q_i - 1].samples()

    @property
    def q_z_samples(self):
        """tensor: Samples from the variational posterior for :math:`z`."""
        return self.model.ps.q_z[self._q_i].samples()

    @q_z_samples.setter
    @_dispatch
    def q_z_samples(self, samples: list):
        # We need a setter, because these won't be trainable through gradients.
        self.model.ps.q_z[self._q_i].samples.delete()
        samples = B.stack(*[B.flatten(x) for x in samples], axis=1)
        self.model.ps.q_z[self._q_i].samples.unbounded(init=samples, visible=False)

    @_dispatch
    def elbo(
        self,
        state: B.RandomState,
        t: B.Numeric,
        y: B.Numeric,
        num_samples: B.Int = 1000,
    ):
        """Fit a mean-field approximation and compute an estimate of the resulting ELBO
        collapsed over :math:`q(z|u)`.

        Args:
            state (random state, optional): Random state.
            t (vector): Locations of observations.
            y (vector): Observations.
            num_samples (int, optional): Number of samples to use. Defaults to `1000`.

        Returns:
            scalar: ELBO.
        """
        _fit_mean_field_ca(self.model, t, y)
        return self.elbo_collapsed_z(state, t, y, num_samples=num_samples)

    @_dispatch
    def elbo(self, t: B.Numeric, y: B.Numeric, num_samples: B.Int = 1000):
        state = B.global_random_state(self.model.dtype)
        state, elbo = self.elbo(state, t, y, num_samples=num_samples)
        B.set_global_random_state(state)
        return elbo

    @_dispatch
    def elbo_gibbs(
        self,
        state: B.RandomState,
        t,
        y,
        u: B.Numeric,
        z: B.Numeric,
        num_samples: B.Int = 5,
    ):
        """Compute an estimate of the structured ELBO ignoring the entropy of
        the optimal :math:`q(u)`.

        Args:
            state (random state): Random state.
            t (vector): Locations of observations.
            y (vector): Observations.
            u (matrix): Current sample for :math:`u`.
            z (matrix): Current sample for :math:`z`.
            num_samples (int, optional): Number of Gibbs samples. Defaults to `5`.

        Returns:
            tuple[random state, scalar, matrix, matrix]: Random state, ELBO, updated
                sample for :math:`u`, and updated sample for :math:`z`.
        """
        ts = self.construct_terms(t, y)
        for _ in range(num_samples):
            state, u = self.q_u_optimal(ts, z).sample(state)
            state, z = self.q_z_optimal(ts, u).sample(state)
        elbo = self.log_Z_u(ts, stop_gradient(u)) + self.p_u.logpdf(stop_gradient(u))
        return state, elbo, u, z

    @_dispatch
    def log_Z_u(self, ts: SimpleNamespace, u: B.Numeric):
        """Compute the normalising constant term of the optimal :math:`q(z|u)`.

        Args:
            ts (:class:`types.SimpleNamespace`): Terms.
            u (tensor): Sample for :math:`u` to use.

        Returns:
            scalar: Normalising constant term.
        """
        q_z = self.q_z_optimal(ts, u)
        quadratic = B.sum(ts.y ** 2) + B.sum(u * B.mm(ts.A_sum, u)) + ts.c_sum
        return 0.5 * (
            -ts.n * B.log(2 * B.pi * self.model.noise)
            - quadratic / self.model.noise
            + B.logdet(self.model.K_z)
            - B.logdet(q_z.prec)
            + B.squeeze(B.iqf(q_z.prec, q_z.lam))
        )

    @_dispatch
    def elbo_collapsed_z(self, state: B.RandomState, t, y, num_samples: B.Int = 1000):
        """Compute an estimate of the ELBO collapsed over :math:`q(z|u)`.

        Args:
            state (random state): Random state.
            t (vector): Locations of observations.
            y (vector): Observations.
            num_samples (int, optional): Number of samples to use. Defaults to `1000`.

        Returns:
            tuple[random state, scalar] : Random state and ELBO.
        """
        ts = self.construct_terms(t, y)
        q_u = self.q_u
        state, us = q_u.sample(state, num_samples)

        @B.jit
        def rec(u):
            return self.log_Z_u(ts, u)

        recs = [rec(u) for u in _columns(us)]
        return state, sum(recs) / len(recs) - q_u.kl(self.p_u)

    def _sample_p_u(self, num_samples):
        if self._q_i == 0:
            return _columns(self.p_u.sample(num_samples))
        else:
            return _columns(self.p_u_samples, num_samples)

    @_dispatch
    def predict(self, t, num_samples: B.Int = 1000):
        """Predict.

        Args:
            t (vector): Points to predict at.
            num_samples (int, optional): Number of samples to use. Defaults to `1000`.

        Returns:
            tuple: Tuple containing the mean and standard deviation of the
                predictions.
        """
        ts = self.construct_terms(t)

        @B.jit
        def predict_moments(u):
            q_z = self.q_z_optimal(self.ts, u)
            return self._predict_moments(ts, u, B.outer(u), q_z.mean, q_z.m2)

        m1s, m2s = zip(*[predict_moments(u) for u in self._sample_p_u(num_samples)])
        m1 = B.mean(B.stack(*m1s, axis=0), axis=0)
        m2 = B.mean(B.stack(*m2s, axis=0), axis=0)
        return m1, m2 - m1 ** 2

    @_dispatch
    def sample_kernel(self, t_k, num_samples: B.Int = 1000):
        """Sample kernel.

        Args:
            t_k (vector): Time point to sample at.
            num_samples (int, optional): Number of samples to use. Defaults to `1000`.

        Returns:
            tensor: Samples.
        """
        us = self._sample_p_u(num_samples)
        sample_kernel = B.jit(self._sample_kernel)
        return B.stack(*[sample_kernel(t_k, u) for u in us], axis=0)

    def _sample_p_z(self, num_samples):
        if self._q_i == 0:
            return _columns(self.p_z.sample(num_samples))
        else:
            return _columns(self.p_z_samples, num_samples)

    @_dispatch
    def predict_z(self, num_samples: B.Int = 1000):
        """Predict Fourier features.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `1000`.

        Returns:
            tuple[vector, vector]: Marginals of the predictions.
        """
        zs = [B.flatten(x) for x in self._sample_p_z(num_samples)]
        m1 = B.mean(B.stack(*zs, axis=0), axis=0)
        m2 = B.mean(B.stack(*[z ** 2 for z in zs], axis=0), axis=0)
        return m1, m2 - m1 ** 2


class MeanField(Approximation):
    """Mean-field approximation of the GPCM.

    Args:
        model (:class:`.gpcm.AbstractGPCM`): Instantiated GPCM.
        fit (str): Fitting method. Must be one of `ca`, `bfgs`, `collapsed-bfgs`.
    """

    def __init__(self, model, fit):
        super().__init__(model)
        self.fit = fit

    @_dispatch
    def elbo(
        self,
        state: B.RandomState,
        t: B.Numeric,
        y: B.Numeric,
        collapsed: Union[None, str] = None,
    ):
        """Compute the mean-field ELBO.

        Args:
            state (random state, optional): Random state.
            t (vector): Locations of observations.
            y (vector): Observations.
            collapsed (str, optional): Collapse over :math:`z` or :math:`u`.

        Returns:
            scalar: ELBO,
        """
        ts = self.construct_terms(t, y)
        if collapsed is None:
            q_u = self.q_u
            q_z = self.q_z
        elif collapsed == "z":
            q_u = self.q_u
            q_z = self.q_z_optimal_mean_field(ts, q_u)
        elif collapsed == "u":
            q_z = self.q_z
            q_u = self.q_u_optimal_mean_field(ts, q_z)
        else:
            raise ValueError(f'Invalid value "{collapsed}" for `collapsed`.')
        return state, (
            (
                -0.5 * ts.n * B.log(2 * B.pi * self.model.noise)
                + (
                    (-0.5 / self.model.noise)
                    * (
                        B.sum(ts.y ** 2)
                        + B.sum(ts.A_sum * q_u.m2)
                        + B.sum(ts.B_sum * q_z.m2)
                        + B.sum(B.mm(ts.I_uz, q_z.m2, ts.I_uz, tr_c=True) * q_u.m2)
                        + ts.c_sum
                        - 2 * B.sum(q_u.mean * B.mm(ts.I_uz_sum, q_z.mean))
                    )
                )
            )
            - q_u.kl(self.p_u)
            - q_z.kl(self.p_z)
        )

    @_dispatch
    def elbo(self, t: B.Numeric, y: B.Numeric, collapsed: Union[None, str] = None):
        state = B.global_random_state(self.model.dtype)
        state, elbo = self.elbo(state, t, y, collapsed=collapsed)
        B.set_global_random_state(state)
        return elbo

    @_dispatch
    def predict(self, t):
        """Predict.

        Args:
            t (vector): Points to predict at.

        Returns:
            tuple[vector, vector]: Marginals of the predictions.
        """
        m1, m2 = self._predict_moments(
            self.construct_terms(t),
            self.p_u.mean,
            self.p_u.m2,
            self.p_z.mean,
            self.p_z.m2,
        )
        return m1, m2 - m1 ** 2

    @_dispatch
    def sample_kernel(self, t_k, num_samples: B.Int = 100):
        """Sample kernel under the mean-field approximation.

        Args:
            t_k (vector): Time point to sample at.
            num_samples (int, optional): Number of samples to use. Defaults to `100`.

        Returns:
            tensor: Samples.
        """
        state = B.global_random_state(self.model.dtype)
        state, us = self.p_u.sample(state, num_samples)
        B.set_global_random_state(state)
        return B.stack(*[self._sample_kernel(t_k, u) for u in _columns(us)], axis=0)

    @_dispatch
    def predict_z(self):
        """Predict Fourier features under the mean-field approximation.

        Returns:
            tuple[vector, vector]: Marginals of the predictions.
        """
        return B.flatten(self.p_z.mean), B.diag(self.p_z.var)


@fit.dispatch
def fit(model, t, y, approximation: Structured, iters: B.Int = 5000):
    """Fit a structured approximation.

    Args:
        model (:class:`.gpcm.AbstractGPCM`): Model.
        approximation (:class:`.Structured`): Approximation.
        t (vector): Locations of observations.
        y (vector): Observations.
        iters (int, optional): Gibbs sampling iterations. Defaults to `5000`.
    """

    def gibbs_sample(state, iters, subsample=None, u=None, z=None):
        """Perform Gibbs sampling."""
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

        # Initialise the state for the Gibbs sampler.
        if u is None:
            state, u = instance.approximation.q_u.sample(state)
        if z is None:
            state, z = instance.approximation.q_z.sample(state)

        # Perform Gibbs sampling.
        with wbml.out.Progress(name="Gibbs sampling", total=iters) as progress:
            us, zs = [], []
            for i in range(iters):
                state, u = sample_u(state, z)
                state, z = sample_z(state, u)
                if subsample is None or i % subsample == 0:
                    us.append(u)
                    zs.append(z)
                progress()

        return state, us, zs

    # Maintain a random state.
    state = B.create_random_state(model.dtype, seed=0)

    # Find a good starting point for the optimisation.
    state, us, zs = gibbs_sample(state, iters=200)  # `200` should be roughly good.
    u, z = us[-1], zs[-1]

    def objective(vs_, state, u, z):
        state, elbo, u, z = model(vs_).approximation.elbo_gibbs(state, t, y, u, z)
        return -elbo, state, u, z

    # Optimise hyperparameters.
    _, state, u, z = minimise_adam(
        objective,
        (model.vs, state, u, z),
        iters=(2 * iters) // 5,  # Spend twice the sampling budget.
        rate=1e-2,
        trace=True,
        jit=True,
        names=model().approximation.ignore_qs(previous=True, current=True),
    )

    # Produce 100 final Gibbs samples and store those samples.
    hunderds = iters // 100
    state, us, zs = gibbs_sample(state, hunderds * 100, subsample=hunderds)
    instance = model()
    instance.approximation.q_u_samples = us
    instance.approximation.q_z_samples = zs


@fit.dispatch
def fit(model, t, y, approximation: MeanField, iters: B.Int = 1000):
    """Fit a mean-field approximation.

    Args:
        model (:class:`.gpcm.AbstractGPCM`): Model.
        approximation (:class:`.MeanField`): Approximation.
        t (vector): Locations of observations.
        y (vector): Observations.
        iters (int, optional): Fixed point iterations. Defaults to `1000`.
    """
    # Maintain a random state.
    state = B.create_random_state(model.dtype, seed=0)

    if approximation.fit == "ca":
        _fit_mean_field_ca(model(), t, y, iters)
    elif approximation.fit == "bfgs":

        def objective(vs_, state):
            state, elbo = model(vs_).approximation.elbo(state, t, y)
            return -elbo, state

        # Optimise hyperparameters.
        _, state = minimise_l_bfgs_b(
            objective,
            (model.vs, state),
            iters=iters,
            trace=True,
            jit=True,
            names=model().approximation.ignore_qs(previous=True),
        )
    elif approximation.fit == "collapsed-bfgs":

        def objective(vs_, state):
            state, elbo = model(vs_).approximation.elbo(state, t, y, collapsed="z")
            return -elbo, state

        # Optimise hyperparameters.
        _, state = minimise_l_bfgs_b(
            objective,
            (model.vs, state),
            iters=iters,
            trace=True,
            jit=True,
            names=model().approximation.ignore_qs(previous=True, current="z"),
        )

        # Explicitly update :math:`q(z)`: it was collapsed in the ELBO.
        instance = model()
        ts = instance.approximation.construct_terms(t, y)
        q_u = instance.approximation.q_u
        q_z = instance.approximation.q_z_optimal_mean_field(ts, q_u)
        instance.approximation.q_z = q_z
    else:
        raise ValueError(f'Invalid value "{approximation.fit}" for `fit`.')
