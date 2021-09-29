import math
from types import SimpleNamespace
from typing import Union

import lab.jax as B
from matrix import Diagonal
from plum import Dispatcher
from probmods import Model
from stheno.jax import Normal

__all__ = ["GPCMApproximation"]

_dispatch = Dispatcher()


@_dispatch
def _m1s_m2s_to_marginals(m1s: Union[list, tuple], m2s: Union[list, tuple]):
    m1 = B.mean(B.stack(*m1s, axis=0), axis=0)
    m2 = B.mean(B.stack(*m2s, axis=0), axis=0)
    return m1, m2 - m1 ** 2


@_dispatch
def _subsample_mc(xs: B.Numeric, n: int):
    if n == 1:
        return xs[:, -1:]
    elif n < 1:
        raise ValueError(f"Invalid number of samples {n}.")

    total = B.shape(xs, 1)

    # Determine which indices to take.
    inds = list(range(total))[:: max(total // n, 1)]
    inds = [i + len(xs) - 1 - inds[-1] for i in inds[-n:]]

    return [xs[:, i : i + 1] for i in inds]


def _mean_var(lam, prec):
    var = B.pd_inv(prec)
    return B.dense(B.cholsolve(B.chol(prec), lam)), var


def _mean_m2(lam, prec):
    chol = B.chol(prec)
    mean = B.cholsolve(chol, lam)
    m2 = prec + B.outer(lam)
    return B.dense(mean), B.cholsolve(chol, B.transpose(B.cholsolve(chol, m2)))


def _sample(state, lam_prec, num_samples):
    lam, prec = lam_prec
    state, noise = Normal(prec).sample(state, num_samples)
    return state, B.cholsolve(B.chol(prec), noise + lam)


def _kl(lam_prec1, lam_prec2):
    lam1, prec1 = lam_prec1
    lam2, prec2 = lam_prec2
    chol1 = B.chol(prec1)
    chol2 = B.chol(prec2)
    diff = B.cholsolve(chol1, lam1) - B.cholsolve(chol2, lam2)
    ratio = B.solve(chol1, chol2)
    return 0.5 * (
        B.sum(ratio ** 2)
        - 2 * B.logdet(ratio)
        + B.sum(B.mm(prec2, diff) * diff)
        - B.cast(B.dtype(lam1, prec1, lam2, prec2), B.shape_matrix(lam1, 0))
    )


def _parametrise_natural(params, n):
    return (
        params.lam.unbounded(B.randn(n, 1), shape=(n, 1)),
        params.prec.positive_definite(1e-1 * B.eye(n), shape=(n, n)),
    )


@_dispatch
def _columns(xs: B.Numeric):
    return [xs[:, i : i + 1] for i in range(B.shape(xs, 1))]


class GPCMApproximation:
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
            return (0, self.model.K_u)
        else:
            return _parametrise_natural(
                self.model.ps.q_u[self._q_i - 1],
                self.model.n_u,
            )

    @property
    def p_u_samples(self):
        """list[tensor]: Samples for the current prior for :math:`u`."""
        return self.model.ps.q_u[self._q_i - 1].samples()

    @property
    def q_u(self):
        """:class:`stheno.Normal`: Current variational posterior for :math:`u`."""
        return _parametrise_natural(self.model.ps.q_u[self._q_i], self.model.n_u)

    @q_u.setter
    @_dispatch
    def q_u(self, lam_prec: tuple):
        lam, prec = lam_prec
        self.model.ps.q_u[self._q_i].lam.assign(B.dense(lam))
        self.model.ps.q_u[self._q_i].prec.assign(B.dense(prec))

    @property
    def q_u_samples(self):
        """list[tensor]: Samples for the current variational posterior for :math:`u`."""
        return self.model.ps.q_u[self._q_i].samples()

    @q_u_samples.setter
    @_dispatch
    def q_u_samples(self, samples: list):
        # We need a setter, because these won't be trainable through gradients.
        self.model.ps.q_u[self._q_i].samples.delete()
        samples = B.stack(*[B.flatten(x) for x in samples], axis=1)
        self.model.ps.q_u[self._q_i].samples.unbounded(init=samples, visible=False)

    @property
    def p_z(self):
        """:class:`stheno.Normal`: Current prior for :math:`z`."""
        if self._q_i == 0:
            return (0, self.model.K_z)
        else:
            return _parametrise_natural(
                self.model.ps.q_z[self._q_i - 1],
                self.model.n_z,
            )

    @property
    def p_z_samples(self):
        """list[tensor]: Samples from the current prior for :math:`z`."""
        return self.model.ps.q_z[self._q_i - 1].samples()

    @property
    def q_z(self):
        """:class:`stheno.Normal`: Variational posterior for :math:`z`."""
        return _parametrise_natural(self.model.ps.q_z[self._q_i], self.model.n_z)

    @q_z.setter
    @_dispatch
    def q_z(self, lam_prec: tuple):
        lam, prec = lam_prec
        self.model.ps.q_z[self._q_i].lam.assign(B.dense(lam))
        self.model.ps.q_z[self._q_i].prec.assign(B.dense(prec))

    @property
    def q_z_samples(self):
        """list[tensor]: Samples from the variational posterior for :math:`z`."""
        return self.model.ps.q_z[self._q_i].samples()

    @q_z_samples.setter
    @_dispatch
    def q_z_samples(self, samples: list):
        # We need a setter, because these won't be trainable through gradients.
        self.model.ps.q_z[self._q_i].samples.delete()
        samples = B.stack(*[B.flatten(x) for x in samples], axis=1)
        self.model.ps.q_z[self._q_i].samples.unbounded(init=samples, visible=False)

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
    def q_u_optimal_natural(self, ts: SimpleNamespace, z):
        """Compute the optimal :math:`q(u|z)`.

        Args:
            ts (object): Terms.
            u (tensor): Sample for :math:`z`.

        Returns:
            tuple[tensor, tensor]: Natural parameters.
        """
        return self._q_optimal_natural(
            ts.A_sum,
            ts.I_uz,
            ts.I_uz_sum,
            self.model.K_u,
            z,
        )

    @_dispatch
    def q_u_optimal_mean_field_natural(self, ts: SimpleNamespace, lam, prec):
        """Compute the optimal :math:`q(u)` in the mean-field approximation.

        Args:
            ts (object): Terms.
            lam (tensor): Lambda of :math:`q(z)`.
            prec (tensor): Precision of :math:`q(z)`.

        Returns:
            tuple[tensor, tensor]: Natural parameters.
        """
        return self._q_optimal_natural(
            ts.A_sum,
            ts.I_uz,
            ts.I_uz_sum,
            self.model.K_u,
            *_mean_m2(lam, prec),
        )

    @_dispatch
    def q_z_optimal_natural(self, ts: SimpleNamespace, u):
        """Compute the optimal :math:`q(z|u)`.

        Args:
            ts (object): Terms.
            u (tensor): Sample for :math:`u`.

        Returns:
            tuple[tensor, tensor]: Natural parameters.
        """
        return self._q_optimal_natural(
            ts.B_sum,
            B.transpose(ts.I_uz),
            B.transpose(ts.I_uz_sum),
            self.model.K_z,
            u,
        )

    @_dispatch
    def q_z_optimal_mean_field_natural(self, ts: SimpleNamespace, lam, prec):
        """Compute the optimal :math:`q(z)` in the mean-field approximation.

        Args:
            ts (object): Terms.
            lam (tensor): Lambda of :math:`q(u)`.
            prec (tensor): Precision of :math:`q(u)`.

        Returns:
            tuple[tensor, tensor]: Natural parameters.
        """
        return self._q_optimal_natural(
            ts.B_sum,
            B.transpose(ts.I_uz),
            B.transpose(ts.I_uz_sum),
            self.model.K_z,
            *_mean_m2(lam, prec),
        )

    @_dispatch
    def _q_optimal_natural(self, A, I, I_sum, K, x, x2=None):
        if x2 is not None:
            inner = B.mm(I, x2, I, tr_c=True)
        else:
            # This is _much_ more efficient!
            part = B.mm(I, x)
            inner = B.mm(part, part, tr_b=True)
        return (
            B.mm(I_sum, x) / self.model.noise,
            K + (A + B.sum(inner, axis=0)) / self.model.noise,
        )

    def log_Z_u(self, ts, u):
        """Compute the normalising constant term of the optimal :math:`q(z|u)`.

        Args:
            ts (object): Terms.
            u (tensor): Sample for :math:`u` to use.

        Returns:
            scalar: Normalising constant term.
        """
        lam, prec = self.q_z_optimal_natural(ts, u)
        quadratic = B.sum(ts.y ** 2) + B.sum(u * B.mm(ts.A_sum, u)) + ts.c_sum
        return 0.5 * (
            -ts.n * B.log(2 * B.pi * self.model.noise)
            - quadratic / self.model.noise
            + B.logdet(self.model.K_z)
            - B.logdet(prec)
            + B.squeeze(B.iqf(prec, lam))
        )

    def log_Z_z(self, ts, z):
        """Compute the normalising constant term of the optimal :math:`q(u|z)`.

        Args:
            ts (object): Terms.
            z (tensor): Sample for :math:`z` to use.

        Returns:
            scalar: Normalising constant term.
        """
        lam, prec = self.q_u_optimal_natural(ts, z)
        quadratic = B.sum(ts.y ** 2) + B.sum(z * B.mm(ts.B_sum, z)) + ts.c_sum
        return 0.5 * (
            -ts.n * B.log(2 * B.pi * self.model.noise)
            - quadratic / self.model.noise
            + B.logdet(self.model.K_u)
            - B.logdet(prec)
            + B.squeeze(B.iqf(prec, lam))
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
        state, us = _sample(state, self.q_u, num_samples)
        recs = [self.log_Z_u(ts, u) for u in _columns(us)]
        return state, sum(recs) / len(recs) - _kl(self.q_u, self.p_u)

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
        state, us = _sample(state, self.q_z, num_samples)
        recs = [self.log_Z_z(ts, z) for z in _columns(zs)]
        return state, sum(recs) / len(recs) - _kl(self.q_z, self.p_z)

    @_dispatch
    def elbo_mean_field(self, t, y):
        """Compute the mean-field ELBO.

        Args:
            t (vector): Locations of observations.
            y (vector): Observations.

        Returns:
            scalar: ELBO.
        """
        ts = self.construct_terms(t, y)
        u, u2 = _mean_m2(*self.q_u)
        z, z2 = _mean_m2(*self.q_z)
        return (
            (
                -0.5 * ts.n * B.log(2 * B.pi * self.model.noise)
                + (
                    (-0.5 / self.model.noise)
                    * (
                        B.sum(ts.y ** 2)
                        + B.sum(ts.A_sum * u2)
                        + B.sum(ts.B_sum * z2)
                        + B.sum(B.mm(ts.I_uz, z2, ts.I_uz, tr_c=True) * u2)
                        + ts.c_sum
                        - 2 * B.sum(u * B.mm(ts.I_uz_sum, z))
                    )
                )
            )
            - _kl(self.q_u, self.p_u)
            - _kl(self.q_z, self.p_z)
        )

    def predict(self, t, num_samples=100):
        """Predict.

        Args:
            t (vector): Points to predict at.
            num_samples (int, optional): Number of samples to use. Defaults to `100`.

        Returns:
            tuple: Tuple containing the mean and standard deviation of the
                predictions.
        """
        ts = self.construct_terms(t)
        m1s, m2s = [], []
        for u in _subsample_mc(self.p_u_samples, num_samples):
            q_z = self.q_z_optimal_natural(self.ts, u)
            m1, m2 = self._predict_moments(
                ts,
                u,
                B.outer(u),
                *_mean_m2(*q_z),
            )
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
            *_mean_m2(*self.p_u),
            *_mean_m2(*self.p_z),
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

    def sample_kernel(self, t_k, num_samples=100):
        """Sample kernel.

        Args:
            t_k (vector): Time point to sample at.
            num_samples (int, optional): Number of samples to use. Defaults to `100`.

        Returns:
            tensor: Samples.
        """
        us = _subsample_mc(self.p_u_samples, num_samples)
        return B.stack(*[self._sample_kernel(t_k, u) for u in us], axis=0)

    def sample_kernel_mean_field(self, t_k, num_samples=100):
        """Sample kernel under the mean-field approximation.

        Args:
            t_k (vector): Time point to sample at.
            num_samples (int, optional): Number of samples to use. Defaults to `100`.

        Returns:
            tensor: Samples.
        """
        state = B.global_random_state(self.model.dtype)
        state, us = _sample(state, self.p_u, num_samples)
        B.set_global_random_state(state)
        return B.stack(*[self._sample_kernel(t_k, u) for u in _columns(us)], axis=0)

    def _sample_kernel(self, t_k, u):
        return B.flatten(
            self.model.kernel_approx(
                t_k,
                B.zero(u)[None],
                B.flatten(B.matmul(self.model.K_u, u)),
            )
        )

    def predict_z(self, num_samples=100):
        """Predict Fourier features.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `100`.

        Returns:
            tuple: Marginals of the predictions.
        """
        samples = [B.flatten(x) for x in _subsample_mc(self.p_z_samples, num_samples)]
        return _m1s_m2s_to_marginals(samples, [x ** 2 for x in samples])

    def predict_z_mean_field(self):
        """Predict Fourier features.

        Returns:
            tuple: Marginals of the predictions.
        """
        return Normal(*_mean_var(*self.p_z)).marginals()
