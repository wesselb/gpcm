from functools import partial

import jax
import jax.numpy as jnp
import lab.jax as B
import wbml.out
from matrix import Diagonal
from plum import Val, Dispatcher
from probmods import Model, instancemethod, priormethod, cast, fit
from stheno.jax import Normal
from varz import Vars, minimise_adam, minimise_l_bfgs_b

from .util import summarise_samples, estimate_psd, closest_psd

__all__ = ["AbstractGPCM"]

_dispatch = Dispatcher()


def _sample(dist, num):
    samples = dist.sample(num=num)
    return [samples[:, i : i + 1] for i in range(num)]


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

        # Construct variational posterior.
        self.q_u = self._parametrise_q_u(self.ps.q_u[0])

    def _parametrise_q_u(self, params):
        diag = Diagonal(params.var.positive(1e-1, shape=(self.n_u,)))
        return Normal(
            params.mean.unbounded(shape=(self.n_u, 1)),
            self.K_u_inv - B.pd_inv(self.K_u + diag),
        )

    @cast
    def __condition__(self, t, y):
        self.p_u = self.q_u
        self.q_u = self._parametrise_q_u(next(self.ps.q_u))
        # TODO: Lazily construct ELBO quantities? Is this call here necessary?
        self._construct_elbo_quantities(t, y)

    @cast
    def _construct_elbo_quantities(self, t, y):
        """Construct quantities required to compute the ELBO.

        Args:
            t (vector): Locations of observations.
            y (vector): Observations.
        """
        # Construct integrals.
        self.I_hx = self.compute_i_hx(t, t)
        self.I_ux = self.compute_I_ux()
        self.I_hz = self.compute_I_hz(t)
        self.I_uz = self.compute_I_uz(t)

        self.n = B.length(t)
        self.y = y

        # Do some precomputations.
        self.I_hx_sum = B.sum(self.I_hx, axis=0)
        self.I_hz_sum = B.sum(self.I_hz, axis=0)
        self.I_ux_sum = self.n * self.I_ux
        self.I_uz_sum = B.sum(y[:, None, None] * self.I_uz, axis=0)  # Weighted by data

        K_u_squeezed = B.mm(self.I_uz, B.dense(self.K_u_inv), self.I_uz, tr_a=True)
        K_z_squeezed = B.mm(self.I_uz, B.dense(self.K_z_inv), self.I_uz, tr_c=True)
        self.A_sum = self.I_ux_sum - B.sum(K_z_squeezed, axis=0)
        self.B_sum = self.I_hz_sum - B.sum(K_u_squeezed, axis=0)
        self.c_sum = (
            self.I_hx_sum
            - B.sum(self.K_u_inv * self.I_ux_sum)
            - B.sum(self.K_z_inv * self.I_hz_sum)
            + B.sum(B.dense(self.K_u_inv) * K_z_squeezed)
        )

    def _optimal_q_z(self, u):
        """Compute the optimal :math:`q(z|u)`.

        Args:
            u (tensor): Sample for :math:`u` to use.

        Returns:
            :class:`stheno.Normal`: Optimal :math:`q(z|u)`.
        """
        part = B.mm(self.I_uz, u, tr_a=True)
        inv_cov_z = self.K_z + 1 / self.noise * (
            self.B_sum + B.sum(B.mm(part, part, tr_b=True), axis=0)
        )
        inv_cov_z_mu_z = 1 / self.noise * B.mm(self.I_uz_sum, u, tr_a=True)
        cov_z = B.pd_inv(inv_cov_z)
        return Normal(B.mm(cov_z, inv_cov_z_mu_z), cov_z)

    def _compute_log_Z(self, u):
        """Compute the normalising constant term.

        Args:
            u (tensor): Sample for :math:`u` to use.

        Returns:
            scalar: Normalising constant term.
        """
        q_z = self._optimal_q_z(u)

        return (
            -0.5 * self.n * B.log(2 * B.pi * self.noise)
            + (
                -0.5
                / self.noise
                * (B.sum(self.y ** 2) + B.sum(u * B.mm(self.A_sum, u)) + self.c_sum)
            )
            + -0.5 * B.logdet(self.K_z)
            + 0.5 * B.logdet(q_z.var)
            + 0.5 * B.iqf(q_z.var, q_z.mean)[0, 0]
        )

    @instancemethod
    @cast
    def logpdf_optimal_q_u(self, t, y, u):
        """Compute the log-pdf of the optimal :math:`q(u)` up to a normalising
        constant.

        Args:
            t (vector): Locations of observations.
            y (vector): Observations.
            u (vector): Value of :math:`u`.

        Returns:
            scalar: Log-pdf.
        """
        self._construct_elbo_quantities(t, y)
        u = B.uprank(u, rank=2)  # Internally, :math:`u` must be a column vector.
        return self._compute_log_Z(u) + self.p_u.logpdf(u)

    @_dispatch
    def elbo(self, t, y, *, num_samples=5):
        state, elbo = self.elbo(
            B.global_random_state(self.dtype), t, y, num_samples=num_samples
        )
        B.set_global_random_state(state)
        return elbo

    @_dispatch
    @instancemethod
    @cast
    def elbo(self, state, t, y, *, num_samples=5):
        """Compute an estimate of the doubly collapsed ELBO.

        Args:
            t (vector): Locations of observations.
            y (vector): Observations.
            num_samples (int, optional): Number of samples to use. Defaults to `5`.

        Returns:
            scalar: ELBO.
        """
        self._construct_elbo_quantities(t, y)
        rec_samples = []
        for _ in range(num_samples):
            state, u = self.q_u.sample(state)
            rec_samples.append(self._compute_log_Z(u))
        return state, sum(rec_samples) / num_samples - self.q_u.kl(self.p_u)

    @instancemethod
    @cast
    def predict(self, t, num_samples=50):
        """Predict.

        Args:
            t (vector): Points to predict at.
            num_samples (int, optional): Number of samples to use. Defaults to `50`.

        Returns:
            tuple: Tuple containing the mean and standard deviation of the
                predictions.
        """
        # Construct integrals for prediction.
        I_hx = self.compute_i_hx(t, t)
        I_ux = self.compute_I_ux()
        I_hz = self.compute_I_hz(t)
        I_uz = self.compute_I_uz(t)

        # Do some precomputations for prediction.
        K_u_squeezed = B.mm(I_uz, B.dense(self.K_u_inv), I_uz, tr_a=True)
        K_z_squeezed = B.mm(I_uz, B.dense(self.K_z_inv), I_uz, tr_c=True)

        m1s = []
        m2s = []

        for u in _sample(self.p_u, num_samples):
            q_z = self._optimal_q_z(u=u)

            # Compute first moment.
            m1 = B.flatten(B.mm(u, I_uz, B.dense(q_z.mean), tr_a=True))

            # Compute second moment.
            A = I_ux - K_z_squeezed + B.mm(I_uz, B.dense(q_z.m2), I_uz, tr_c=True)
            B_ = I_hz - K_u_squeezed
            c = (
                I_hx
                - B.sum(self.K_u_inv * I_ux)
                - B.sum(B.dense(self.K_z_inv) * I_hz, axis=(1, 2))
                + B.sum(B.dense(self.K_u_inv) * K_z_squeezed, axis=(1, 2))
            )
            m2 = (
                B.sum(A * B.outer(u), axis=(1, 2))
                + B.sum(B_ * B.dense(q_z.m2), axis=(1, 2))
                + c
            )

            m1s.append(m1)
            m2s.append(m2)

        # Estimate mean and variance.
        m1 = B.mean(B.stack(*m1s, axis=0), axis=0)
        m2 = B.mean(B.stack(*m2s, axis=0), axis=0)
        return m1, m2 - m1 ** 2

    @instancemethod
    def predict_kernel(self, num_samples=200):
        """Predict kernel and normalise prediction.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `200`.

        Returns:
            :class:`collections.namedtuple`: The prediction.
        """
        return summarise_samples(*self.predict_kernel_samples(num_samples=num_samples))

    @instancemethod
    def predict_kernel_samples(self, num_samples=200):
        """Predict kernel and normalise prediction.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `200`.

        Returns:
            tuple[vector, tensor]: Tuple containing the inputs of the samples and the
                samples.
        """
        # Sample kernels.
        ks = []
        t_k = B.linspace(self.dtype, 0, 1.2 * B.max(self.t_u), 300)
        for u in _sample(self.p_u, num_samples):
            k = self.kernel_approx(
                t_k, B.zeros(self.dtype, 1), u=B.flatten(B.dense(B.matmul(self.K_u, u)))
            )
            ks.append(B.flatten(k))
        ks = B.stack(*ks, axis=0)

        # Normalise predicted kernel.
        var_mean = B.mean(ks[:, 0])
        ks = ks / var_mean
        wbml.out.kv("Mean variance of kernel prediction", var_mean)

        return t_k, ks

    @instancemethod
    def predict_psd(self, num_samples=200):
        """Predict the PSD in dB.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `200`.

        Returns:
            :class:`collections.namedtuple`: Predictions.
        """
        t_k, ks = self.predict_kernel_samples(num_samples=num_samples)

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
        m1s = []
        m2s = []

        for u in _sample(self.p_u, num_samples):
            q_z = self._optimal_q_z(u=u)
            m1s.append(B.flatten(B.dense(q_z.mean)))
            m2s.append(B.diag(q_z.var) + m1s[-1] ** 2)

        # Estimate mean and associated error bounds.
        m1 = B.mean(B.stack(*m1s, axis=0), axis=0)
        m2 = B.mean(B.stack(*m2s, axis=0), axis=0)
        return m1, m2 - m1 ** 2

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
        L_u = B.cholesky(self.compute_K_u())
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


def laplace_approximation(f, x_init, f_eval=None):
    """Perform a Laplace approximation of a density.

    Args:
        f (function): Possibly unnormalised log-density.
        x_init (column vector): Starting point to start the optimisation.
        f_eval (function): Use this log-density for the evaluation at the MAP estimate.

    Returns:
        tuple[:class:`stheno.Normal`]: Laplace approximation.
    """

    def laplace_objective(vs_):
        return -f(vs_["x"])

    # Perform optimisation.
    vs = Vars(B.dtype(x_init))
    vs.unbounded(init=x_init, name="x")
    minimise_l_bfgs_b(laplace_objective, vs, iters=2_000, jit=True, trace=True)
    x = vs["x"]

    # Compute Laplace approximation.
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
        iters (int, optional): Training iterations. Defaults to `100`.
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
    minimise_adam(
        objective,
        (model.vs, state),
        iters=iters,
        trace=True,
        rate=5e-2,
        names=names,
    )
