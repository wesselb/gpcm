import pickle

import jax.numpy as jnp
import lab.jax as B
import wbml.out
from plum import Dispatcher
from probmods import Model, cast, fit, instancemethod, priormethod
from stheno.jax import Normal
from varz import Vars
from stheno import GP

from .approx import MeanField, Structured
from .util import (
    closest_psd,
    estimate_psd,
    summarise_samples,
    min_phase as transform_min_phase,
)

__all__ = ["AbstractGPCM"]

_dispatch = Dispatcher()


class AbstractGPCM(Model):
    """GPCM model.

    Args:
        scheme (str, optional): Approximation scheme. Must be one of `structured`,
            `mean-field-ca`, `mean-field-gradient`, `mean-field-collapsed-gradient`,
            `mean-field-ca-gradient`, or `mean-field-ca-collapsed-gradient`.
            Defaults to `structured`.
    """

    @_dispatch
    def __init__(self, scheme: str = "structured"):
        self.vs = Vars(jnp.float64)
        self.scheme = scheme.lower()

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
        if self.scheme == "structured":
            self.approximation = Structured(self)
        elif self.scheme == "mean-field":
            # Use the best mean-field scheme.
            self.approximation = MeanField(self, fit="ca-collapsed-bfgs")
        elif self.scheme == "mean-field-ca":
            self.approximation = MeanField(self, fit="ca")
        elif self.scheme == "mean-field-gradient":
            self.approximation = MeanField(self, fit="bfgs")
        elif self.scheme == "mean-field-collapsed-gradient":
            self.approximation = MeanField(self, fit="collapsed-bfgs")
        elif self.scheme == "mean-field-ca-gradient":
            self.approximation = MeanField(self, fit="ca-bfgs")
        elif self.scheme == "mean-field-ca-collapsed-gradient":
            self.approximation = MeanField(self, fit="ca-collapsed-bfgs")
        else:
            raise ValueError(
                f'Invalid value "{self.scheme}" for the approximation scheme.'
            )

    def __condition__(self, t, y):
        self.approximation.condition(t, y)

    @instancemethod
    @cast
    def elbo(self, *args, **kw_args):
        return self.approximation.elbo(*args, **kw_args)

    @instancemethod
    @cast
    def predict(self, *args, **kw_args):
        return self.approximation.predict(*args, **kw_args)

    @instancemethod
    def predict_kernel(self, t_k=None, num_samples=1000):
        """Predict kernel and normalise prediction.

        Args:
            t_k (vector, optional): Inputs to sample kernel at. Will be automatically
                determined if not given.
            num_samples (int, optional): Number of samples to use. Defaults to `1000`.

        Returns:
            :class:`collections.namedtuple`: The prediction.
        """
        return summarise_samples(*self.sample_kernel(t_k=t_k, num_samples=num_samples))

    @instancemethod
    def sample_kernel(self, t_k=None, num_samples=1000):
        """Predict kernel and normalise prediction.

        Args:
            t_k (vector, optional): Inputs to sample kernel at. Will be automatically
                determined if not given.
            num_samples (int, optional): Number of samples to use. Defaults to `1000`.

        Returns:
            tuple[vector, tensor]: Tuple containing the inputs of the samples and the
                samples.
        """
        if t_k is None:
            t_k = B.linspace(self.dtype, 0, self.extent, 300)

        ks = self.approximation.sample_kernel(t_k, num_samples=num_samples)

        # Normalise predicted kernel.
        var_mean = B.mean(ks[:, 0])
        wbml.out.kv("Mean variance of kernel samples", var_mean)

        return t_k, ks

    @instancemethod
    def predict_psd(self, t_k=None, num_samples=1000):
        """Predict the PSD in dB.

        Args:
            t_k (vector, optional): Inputs to sample kernel at. Will be automatically
                determined if not given.
            num_samples (int, optional): Number of samples to use. Defaults to `1000`.

        Returns:
            :class:`collections.namedtuple`: Predictions.
        """
        t_k, ks = self.sample_kernel(t_k, num_samples=num_samples)

        # Estimate PSDs.
        freqs, psds = zip(*[estimate_psd(t_k, k, db=False) for k in ks])
        freqs = freqs[0]
        psds = B.stack(*psds, axis=0)

        return summarise_samples(freqs, psds, db=True)

    @instancemethod
    def predict_fourier(self, num_samples=1000):
        """Predict Fourier features.

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to `1000`.

        Returns:
            tuple: Marginals of the predictions.
        """
        return self.approximation.predict_z(num_samples=num_samples)

    @instancemethod
    def predict_filter(self, t_h=None, num_samples=1000, min_phase=True):
        """Predict the learned filter.

        Args:
            t_h (vector, optional): Inputs to sample filter at.
            num_samples (int, optional): Number of samples to use. Defaults to `1000`.
            min_phase (bool, optional): Predict a minimum-phase version of the filter.
                Defaults to `True`.

        Returns:
            :class:`collections.namedtuple`: Predictions.
        """
        if t_h is None:
            t_h = B.linspace(
                self.dtype,
                0 if min_phase else -self.extent,
                self.extent,
                601
            )

        @B.jit
        def sample_h(state):
            state, u = self.approximation.p_u.sample(state)
            u = B.mm(self.K_u, u)  # Transform :math:`\hat u` into :math:`u`.
            h = GP(self.k_h())
            h = h | (h(self.t_u), u)  # Condition on sample.
            state, h = h(t_h).sample(state)  # Sample at desired points.
            return state, B.flatten(h)

        # Perform sampling.
        state = B.global_random_state(self.dtype)
        samples = []
        for _ in range(num_samples):
            state, h = sample_h(state)

            # Transform sample according to specification.
            if min_phase:
                h = transform_min_phase(h)

            samples.append(h)
        B.set_global_random_state(state)

        return summarise_samples(t_h, B.stack(*samples, axis=0))

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
        return K, y

    def save(self, path):
        """Save model and inference results to a file.

        Args:
            path (str): Path to save to.
        """
        data = {name: B.to_numpy(B.dense(self.vs[name])) for name in self.vs.names}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        """Load model from a file.

        Args:
            path (str): Path to load from.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        for name, value in data.items():
            if name in self.vs:
                # Overwrite existing values.
                self.vs.assign(name, value)
            else:
                # Assign an invisible bounded variable: we lost the information about
                # the constraints.
                self.vs.unbounded(init=value, visible=False, name=name)


@fit.dispatch
def fit(model: AbstractGPCM, t, y, **kw_args):
    fit(model, t, y, model().approximation, **kw_args)
