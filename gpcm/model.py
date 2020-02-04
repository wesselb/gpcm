import lab as B
import numpy as np
import torch
import torch.autograd
import wbml.out
import wbml.out
from matrix import Dense
from stheno import Normal
from varz import Vars
from varz.torch import minimise_adam, minimise_l_bfgs_b

from .util import summarise_samples, pd_inv, estimate_psd

__all__ = ['Model', 'train']


def _sample(dist, num):
    samples = dist.sample(num=num)
    return [samples[:, i:i + 1] for i in range(num)]


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
        """Compute quantities related to the optimal :math:`q(z|u)`.

        Args:
            u (tensor): Sample for :math:`u` to use.

        Returns:
            tuple: Two-tuple.
        """
        part = B.mm(self.I_uz, u, tr_a=True)
        inv_cov_z = self.K_z + 1/self.noise* \
                    B.sum(self.B_sum + B.mm(part, part, tr_b=True), axis=0)
        inv_cov_z_mu_z = 1/self.noise*B.mm(self.I_uz_sum, u, tr_a=True)
        L_inv_cov_z = B.chol(Dense(inv_cov_z))
        root = B.trisolve(L_inv_cov_z, inv_cov_z_mu_z)

        return L_inv_cov_z, root

    def logpdf_optimal_u(self, u):
        """Compute the log-pdf of the optimal :math:`q(u)` up to a normalising
        constant.

        Args:
            u (tensor): Value of :math:`u`.

        Returns:
            scalar: Log-pdf.
        """
        return self._compute_log_Z(u) + self.p_u.logpdf(u)

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
                                B.sum(u*B.mm(self.A_sum, u)) +
                                self.c_sum) + \
               -0.5*B.logdet(self.K_z) + \
               -0.5*2*B.sum(B.log(B.diag(L_inv_cov_z))) + \
               0.5*B.sum(root**2)

    def elbo(self, q_u, num_samples=200, entropy=True):
        """Compute an estimate of the doubly collapsed ELBO.

        Args:
            q_u (distribution): Distribution :math:`q(u)`.
            num_samples (int, optional): Number of samples to use. Defaults
                to `200`.
            entropy (bool, optional): Also estimate entropy by fitting a
                Gaussian. Defaults to `True`.

        Returns:
            scalar: ELBO.
        """

        # Estimate ELBO without the entropy term.
        elbo = 0
        for u in _sample(q_u, num_samples):
            elbo = elbo + self._compute_log_Z(u) + self.p_u.logpdf(u)
        elbo = elbo/num_samples

        # Add in the entropy term.
        if entropy:
            elbo = elbo + q_u.entropy()

        return elbo

    def predict(self, q_u, num_samples=200):
        """Predict at the data.

        Args:
            q_u (distribution): Distribution :math:`q(u)`.
            num_samples (int, optional): Number of samples to use. Defaults
                to `200`.

        Returns:
            tuple: Tuple containing the mean and standard deviation of the
                predictions.
        """
        means = []
        stds = []

        for u in _sample(q_u, num_samples):
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

    def predict_kernel(self, q_u, num_samples=200, return_samples=False):
        """Predict kernel and normalise prediction.

        Args:
            q_u (distribution): Distribution :math:`q(u)`.
            num_samples (int, optional): Number of samples to use. Defaults
                to `200`.
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
        for u in _sample(q_u, num_samples):
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

    def predict_psd(self, q_u, num_samples=200):
        """Predict the PSD in dB.

        Args:
            q_u (distribution): Distribution :math:`q(u)`.
            num_samples (int, optional): Number of samples to use. Defaults
                to `200`.

        Returns:
            :class:`collections.namedtuple`: Predictions.
        """
        t_k, ks = self.predict_kernel(q_u,
                                      num_samples=num_samples,
                                      return_samples=True)

        # Estimate PSDs.
        freqs, psds = zip(*[estimate_psd(t_k, k) for k in ks])
        freqs = freqs[0]
        psds = B.stack(*psds, axis=0)

        return summarise_samples(freqs, psds)

    def predict_fourier(self, q_u, num_samples=200):
        """Predict Fourier features.

        Args:
            q_u (distribution): Distribution :math:`q(u)`.
            num_samples (int, optional): Number of samples to use. Defaults
                to `200`.

        Returns:
            tuple: Marginals of the predictions.
        """
        means = []
        stds = []

        for u in _sample(q_u, num_samples):
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


def hessian(f, x):
    """Compute the Hessian of a function at a certain input.

    Args:
        f (function): Function to compute Hessian of.
        x (column vector): Input to compute Hessian at.

    Returns:
        matrix: Hessian.
    """
    if B.rank(x) != 2 or B.shape(x)[1] != 1:
        raise ValueError('Input must be a column vector.')

    rows = []
    for i in range(B.shape(x)[0]):
        x_clone = x.detach().requires_grad_(True)
        grad = torch.autograd.grad(f(x_clone), x_clone, create_graph=True)[0]
        grad[i].backward()
        rows.append(x_clone.grad)

    # Assemble and symmetrise.
    hess = B.concat(*rows, axis=1)
    return (hess + B.transpose(hess))/2


def is_pd(x):
    """Check if a matrix is positive definite.

    Args:
        x (matrix): Matrix to check definiteness of.

    Returns:
        bool: `True` if `x` is positive definite and `False` otherwise.
    """
    x = B.to_numpy(x)
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.LinAlgError:
        return False


def laplace_approximation(f, x_init):
    """Perform a Laplace approximation of a density.

    Args:
        f (function): Possibly unnormalised log-density.
        x_init (column vector): Starting point to start the optimisation.

    Returns:
        tuple[:class:`stheno.Normal`, column vector]: Laplace approximation
            and end point of the optimisation.
    """
    vs = Vars(torch.float64)
    vs.get(init=x_init, name='x')

    while True:
        minimise_l_bfgs_b(lambda vs_: -f(vs_['x']), vs, iters=2000)
        x = vs['x']
        precision = -2*hessian(f, x)

        # Regularise the precision to improve stability and inaccuracy due to
        # the gradient not entirely being zero.
        precision = B.reg(precision, diag=1e-6)

        if is_pd(precision):
            return Normal(pd_inv(precision), x), x
        else:
            wbml.out.out('Laplace approximation failed... '
                         'Rerunning optimisation.')


def train(construct_model,
          vs,
          iters=100,
          fix_noise=False):
    """Train a model.

    Args:
        construct_model (function): Function that takes in a variable container
            and gives back the model.
        vs (:class:`varz.Vars`): Variable container.
        iters (int, optional): Training iterations. Defaults to `100`.
        fix_noise (bool, optional): Fix the noise during training. Defaults
            to `False`.

    Returns:
        :class:`stheno.Normal`: Approximate posterior.
    """
    # Persistent state during optimisation.
    state = {'sampler': None, 'u': None}

    def objective(vs_):
        model = construct_model(vs_)

        # Create a detached model for efficiency.
        model_detached = construct_model(vs_.copy(detach=True))

        # Initialise starting point.
        if state['u'] is None:
            state['u'] = B.randn(torch.float64, model_detached.n_u, 1)

        # Perform Laplace approximation..
        dist, state['u'] = \
            laplace_approximation(model_detached.logpdf_optimal_u, state['u'])

        return -model.elbo(dist, entropy=False, num_samples=20)

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

    # Create final Laplace approximation to return.
    model_detached = construct_model(vs.copy(detach=True))
    dist, _ = laplace_approximation(model_detached.logpdf_optimal_u,
                                    state['u'])

    return dist
