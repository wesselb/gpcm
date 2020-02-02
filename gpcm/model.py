import lab as B
import torch
import wbml.out
import wbml.out
from matrix import Dense
from stheno import Normal
from varz.torch import minimise_adam

from gpcm.sample import ESS
from .util import summarise_samples, pd_inv, estimate_psd

__all__ = ['Model', 'train']


class GradientControl:
    """Context manager to potentially disallow gradient computation.

    Args:
        gradient (bool): Allow gradient computation.
    """

    def __init__(self, gradient):
        self.gradient = gradient
        self.context = torch.no_grad()

    def __enter__(self):
        if not self.gradient:
            self.context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.gradient:
            self.context.__exit__(exc_type, exc_val, exc_tb)


class Model:
    """GPCM model."""

    sampler = None

    def __init__(self):
        # Initialise quantities to compute.
        self.n = None
        self.t = None
        self.y = None

        self.K_z = None
        self.K_z_inv = None

        self.K_u = None
        self.K_u_inv = None

        self.I_hx = None
        self.I_ux = None
        self.I_hz = None
        self.I_uz = None

        self.I_uz_sum = None

        self.I_zu_inv_K_u_I_uz = None

        self.A_sum = None
        self.B_sum = None
        self.c_sum = None

        self.L_inv_cov_z = None
        self.root = None

        self.p_u = None

        # Initialise quantities that the particular model should populate.
        self.noise = None
        self.dtype = None
        self.t_u = None

    def construct(self, t, y):
        """Construct quantities for the model.

        Args:
            t (vector): Locations of observations.
            y (vector): Observations.

        Returns:
            :class:`model.Model`: Itself.
        """
        # Construct integrals.
        I_hx = self.compute_i_hx(t, t)
        I_ux = self.compute_I_ux()
        I_hz = self.compute_I_hz(t)
        I_uz = self.compute_I_uz(t)

        # Construct kernel matrices.
        K_z = self.compute_K_z()
        K_z_inv = pd_inv(K_z)

        K_u = self.compute_K_u()
        K_u_inv = pd_inv(K_u)
        L_u = B.chol(K_u)  # Cholesky will be cached.

        n = B.length(t)

        # Do some precomputations.
        L_u_tiled = B.tile(L_u[None, :, :], n, 1, 1)
        L_u_inv_I_uz = B.trisolve(L_u_tiled, I_uz)
        I_zu_inv_K_u_I_uz = B.mm(L_u_inv_I_uz, L_u_inv_I_uz, tr_a=True)
        I_zu_inv_K_u_I_uz_sum = B.sum(I_zu_inv_K_u_I_uz, axis=0)

        I_hx_sum = B.sum(I_hx, axis=0)
        I_hz_sum = B.sum(I_hz, axis=0)
        I_ux_sum = n*I_ux
        I_uz_sum = B.sum(y[:, None, None]*I_uz, axis=0)  # Weighted by data.

        A_sum = I_ux_sum - \
                B.sum(B.mm(I_uz, B.dense(K_z_inv), I_uz, tr_c=True), axis=0)
        B_sum = I_hz_sum - I_zu_inv_K_u_I_uz_sum
        c_sum = I_hx_sum - \
                B.sum(K_u_inv*I_ux_sum) - \
                B.sum(K_z_inv*I_hz_sum) + \
                B.sum(K_z_inv*I_zu_inv_K_u_I_uz_sum)

        # Construct prior p(u).
        p_u = Normal(K_u_inv)

        # Store computed quantities.
        self.n = n
        self.t = t
        self.y = y

        self.K_z = K_z
        self.K_z_inv = K_z_inv

        self.K_u = K_u
        self.K_u_inv = K_u_inv

        self.I_hx = I_hx
        self.I_ux = I_ux
        self.I_hz = I_hz
        self.I_uz = I_uz

        self.I_uz_sum = I_uz_sum

        self.I_zu_inv_K_u_I_uz = I_zu_inv_K_u_I_uz

        self.A_sum = A_sum
        self.B_sum = B_sum
        self.c_sum = c_sum

        self.p_u = p_u

        return self

    def _compute_q_z_related(self, u):
        """Compute quantities related to the optimal :math:`q(z)`.

        Args:
            u (tensor): Sample for :math:`u` to use.

        Returns:
            tuple: Two-tuple.
        """
        inv_cov_z = self.K_z + 1/self.noise* \
                    B.sum(self.B_sum +
                          B.mm(self.I_uz, B.dense(B.outer(u)), self.I_uz,
                               tr_a=True), axis=0)
        inv_cov_z_mu_z = 1/self.noise*B.mm(self.I_uz_sum, u, tr_a=True)
        L_inv_cov_z = B.chol(Dense(inv_cov_z))
        root = B.trisolve(L_inv_cov_z, inv_cov_z_mu_z)

        return L_inv_cov_z, root

    def _compute_log_Z(self, u):
        """Compute the normalising constant term.

        Args:
            u (tensor): Sample for :math:`u` to use.

        Returns:
            scalar: Normalising constant term.
        """
        L_inv_cov_z, root = self._compute_q_z_related(u)

        return -0.5*self.n*B.log(2*B.pi*self.noise) + \
               -0.5/self.noise*(B.sum(self.y**2) +
                                B.sum(B.outer(u)*self.A_sum) +
                                self.c_sum) + \
               -0.5*B.logdet(self.K_z) + \
               -0.5*2*B.sum(B.log(B.diag(L_inv_cov_z))) + \
               0.5*B.sum(root**2)

    def elbo(self, samples, entropy=True):
        """Compute an estimate of the doubly collapsed ELBO.

        Args:
            samples (list[tensor]): Samples for :math:`u`.
            entropy (bool, optional): Also estimate entropy by fitting a
                Gaussian. Defaults to `True`.

        Returns:
            scalar: ELBO.
        """
        num_samples = len(samples)

        # Estimate ELBO without the entropy term.
        elbo = 0
        for u in samples:
            elbo = elbo + self._compute_log_Z(u) + self.p_u.logpdf(u)
        elbo = elbo/num_samples

        # Add in the entropy term.
        if entropy:
            # Estimate :math:`q(u)` by fitting a Gaussian.
            samples = B.concat(*samples, axis=1)
            mean = B.mean(samples, axis=1)[:, None]
            cov = B.matmul(samples - mean, samples - mean,
                           tr_b=True)/num_samples
            q_u = Normal(cov, mean)

            elbo = elbo + q_u.entropy()

        return elbo

    def predict(self, samples):
        """Predict at the data.

        Args:
            samples (list[tensor]): Samples for :math:`u`.

        Returns:
            tuple: Tuple containing the mean and standard deviation of the
                predictions.
        """
        means = []
        stds = []

        for u in samples:
            L_inv_cov_z, root = self._compute_q_z_related(u=u)

            # Use sample for "first and second moment".
            m1 = u
            m2 = B.outer(u)

            # Construct optimal :math:`q(z)`.
            mu_z = B.trisolve(B.transpose(L_inv_cov_z), root, lower_a=False)
            cov_z = B.cholsolve(L_inv_cov_z, B.eye(L_inv_cov_z))
            q_z = Normal(cov_z, mu_z)

            # Compute mean.
            mean = B.flatten(B.mm(m1, self.I_uz, B.dense(mu_z), tr_a=True))

            # Compute variance.
            A = self.I_ux[None, :, :] - \
                B.mm(self.I_uz, B.dense(self.K_z_inv), self.I_uz, tr_c=True)
            B_ = self.I_hz - self.I_zu_inv_K_u_I_uz
            c = self.I_hx - \
                B.sum(self.K_u_inv*self.I_ux) - \
                B.sum(B.sum(self.K_z_inv[None, :, :]*self.I_hz, axis=1),
                      axis=1) + \
                B.sum(B.sum(self.K_z_inv[None, :, :]*self.I_zu_inv_K_u_I_uz,
                            axis=1), axis=1)
            var = B.sum(B.sum(A*m2[None, :, :], axis=1), axis=1) + \
                  B.sum(B.sum(B_*q_z.m2[None, :, :], axis=1), axis=1) + c

            means.append(mean)
            stds.append(B.sqrt(var))

        # Estimate mean and standard deviation.
        mean = B.mean(B.stack(*means, axis=0), axis=0)
        std = B.mean(B.stack(*stds, axis=0), axis=0)

        return mean, std

    def predict_kernel(self, samples, return_samples=False):
        """Predict kernel and normalise prediction.

        Args:
            samples (list[tensor]): Samples for :math:`u`.
            return_samples (bool, optional): Return samples instead of
                prediction. Defaults to `False`.

        Returns:
            :class:`collections.namedtuple` or tuple: Named tuple containing
                the prediction, or a tuple containing the time points of the
                samples and the samples.
        """
        # Sample kernels.
        ks = []
        t_k = B.linspace(self.dtype, 0, 1.2*B.max(self.t_u), 300)
        for u in samples:
            k = self.kernel_approx(t_k, B.zeros(self.dtype, 1),
                                   u=B.flatten(B.dense(B.matmul(self.K_u, u))))
            ks.append(B.flatten(k))
        ks = B.stack(*ks, axis=0)

        # Normalise predicted kernel.
        var_mean = B.mean(ks[:, 0])
        ks = ks/var_mean
        wbml.out.kv('Mean variance of kernel prediction', var_mean)

        if return_samples:
            return t_k, ks
        else:
            return summarise_samples(t_k, ks)

    def predict_psd(self, samples):
        """Predict the PSD in dB.

        Args:
            samples (list[tensor]): Samples for :math:`u`.

        Returns:
            :class:`collections.namedtuple`: Predictions.
        """
        t_k, ks = self.predict_kernel(samples, return_samples=True)

        # Estimate PSDs.
        freqs, psds = zip(*[estimate_psd(t_k, k) for k in ks])
        freqs = freqs[0]
        psds = B.stack(*psds, axis=0)

        return summarise_samples(freqs, psds)

    def predict_fourier(self, samples):
        """Predict Fourier features.

        Args:
            samples (list[tensor]): Samples for :math:`u`.

        Returns:
            tuple: Marginals of the predictions.
        """
        means = []
        stds = []

        for u in samples:
            L_inv_cov_z, root = self._compute_q_z_related(u=u)

            mu_z = B.trisolve(B.transpose(L_inv_cov_z), root, lower_a=False)
            cov_z = B.cholsolve(L_inv_cov_z, B.eye(L_inv_cov_z))

            means.append(B.flatten(B.dense(mu_z)))
            stds.append(B.sqrt(B.diag(cov_z)))

        # Estimate mean and standard deviation.
        mean = B.mean(B.stack(*means, axis=0), axis=0)
        std = B.mean(B.stack(*stds, axis=0), axis=0)

        return mean, mean - 2*std, mean + 2*std

    def kernel_approx(self, t1, t2, u):
        """Kernel approximation using inducing variables :math:`u` for the
        impulse response :math:`h`.

        Args:
            t1 (vector): First time input.
            t2 (vector): Second time input.
            u (vector): Values of the inducing variables.

        Returns:
            tensor: Approximation of the kernel matrix broadcasted over `t1`
                and `t2`.
        """
        # Construct the first part.
        part1 = self.compute_i_hx(t1[:, None], t2[None, :])

        # Construct the second part.
        L_u = B.cholesky(self.compute_K_u())
        inv_L_u = B.dense(B.trisolve(L_u, B.eye(L_u)))
        prod = B.mm(inv_L_u, u[:, None])
        I_ux = self.compute_I_ux(t1, t2)
        trisolved = B.mm(inv_L_u, I_ux, inv_L_u, tr_c=True)
        part2 = \
            B.trace(trisolved, axis1=2, axis2=3) - \
            B.trace(B.mm(prod, trisolved, prod, tr_a=True), axis1=2, axis2=3)

        return part1 - part2

    def sample(self, t, normalise=True):
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
            K = K/K[0, 0]
        f = B.sample(B.reg(K))[:, 0]
        return K, f


def train(construct_model,
          vs,
          burn=300,
          iters=100,
          elbo_burn=20,
          elbo_num_samples=5,
          fix_noise=False):
    """Train a model with the SMF approximation.

    Args:
        construct_model (function): Function that takes in a variable container
            and gives back the model.
        vs (:class:`varz.Vars`): Variable container.
        burn (int, optional): Number of samples to burn before training.
            Defaults to `300`.
        iters (int, optional): Training iterations. Defaults to `100`.
        elbo_burn (int, optional): Number of samples to burn for convergence
            of the sampler to adjust to new hyperparameters. Defaults
            to `20`.
        elbo_num_samples (int, optional): Number of samples to use to estimate
            the ELBO. Defaults to `5`.
        fix_noise (bool, optional): Fix the noise during training. Defaults
            to `False`.

    Returns:
        :class:`.sample.ESS`: Sampler.
    """
    state = {'sampler': None}  # Persisting state.

    def objective(vs_):
        model = construct_model(vs_)

        def log_lik(u):
            with GradientControl(gradient=False):
                return model._compute_log_Z(u)

        def sample_prior():
            with GradientControl(gradient=False):
                return model.p_u.sample()

        # Ensure that the sampler is initialised.
        if state['sampler'] is None:
            state['sampler'] = ESS(log_lik, sample_prior)

            # Burn the sampler to get it into a good position.
            with wbml.out.Section('Burning sampler'):
                state['sampler'].sample(num=burn, trace=True)
        else:
            state['sampler'].log_lik = log_lik
            state['sampler'].sample_prior = sample_prior

        # Obtain samples.
        state['sampler'].sample(num=elbo_burn)
        samples = state['sampler'].sample(num=elbo_num_samples)

        return -model.elbo(samples, entropy=False)

    # Determine the names of the variables to optimise.
    names = vs.names
    if fix_noise:
        names = list(set(names) - {'noise'})

    # Perform optimisation.
    minimise_adam(objective,
                  vs,
                  iters=iters,
                  trace=True,
                  rate=5e-2,
                  names=names)

    return state['sampler']
