import lab as B
import numpy as np
import wbml.out
from matrix import Dense
from stheno import Normal

from .util import collect, pd_inv

__all__ = ['Model']


class Model:
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
        self.q_u = None

    def construct(self, t, y):
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

        # Compute optimal q(z).
        inv_cov_z = \
            K_z + 1/self.noise* \
            B.sum(B_sum + B.mm(I_uz, B.dense(self.q_u.m2), I_uz, tr_a=True),
                  axis=0)
        inv_cov_z_mu_z = 1/self.noise*B.mm(I_uz_sum, self.q_u.mean, tr_a=True)
        L_inv_cov_z = B.chol(Dense(inv_cov_z))
        root = B.trisolve(L_inv_cov_z, inv_cov_z_mu_z)

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

        self.I_zu_inv_K_u_I_uz = I_zu_inv_K_u_I_uz

        self.A_sum = A_sum
        self.B_sum = B_sum
        self.c_sum = c_sum

        self.L_inv_cov_z = L_inv_cov_z
        self.root = root

        self.p_u = p_u

    def elbo(self):
        """Compute the ELBO.

        Returns:
            scalar: ELBO.
        """
        return -0.5*self.n*B.log(2*B.pi*self.noise) + \
               -0.5/self.noise*(B.sum(self.y**2) +
                                B.sum(self.q_u.m2*self.A_sum) +
                                self.c_sum) + \
               -0.5*B.logdet(self.K_z) + \
               -0.5*2*B.sum(B.log(B.diag(self.L_inv_cov_z))) + \
               0.5*B.sum(self.root**2) + \
               -self.q_u.kl(self.p_u)

    def predict(self):
        """Predict at the data.

        Returns:
            tuple: Tuple containing the mean and standard deviation of the
                predictions.
        """

        # Construct optimal q(z).
        mu_z = B.trisolve(B.transpose(self.L_inv_cov_z),
                          self.root, lower_a=False)
        cov_z = B.cholsolve(self.L_inv_cov_z, B.eye(self.L_inv_cov_z))
        q_z = Normal(cov_z, mu_z)

        # Compute mean.
        mu = B.flatten(B.mm(self.q_u.mean, self.I_uz, B.dense(mu_z),
                            tr_a=True))

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
        var = B.sum(B.sum(A*self.q_u.m2[None, :, :], axis=1), axis=1) + \
              B.sum(B.sum(B_*q_z.m2[None, :, :], axis=1), axis=1) + c

        return B.to_numpy(mu), B.to_numpy(B.sqrt(var))

    def predict_kernel(self):
        """Predict kernel and normalise prediction.

        Returns:
            :class:`collections.namedtuple`: Named tuple containing the
                prediction.
        """
        # Transform back to normal space.
        q_u = self.q_u.lmatmul(self.K_u)

        ks = []
        t_k = B.linspace(self.dtype, 0, 1.2*B.max(self.t_u), 100)
        for i in range(100):
            k = self.kernel_approx(t_k, B.zeros(self.dtype, 1),
                                   u=B.flatten(q_u.sample()))
            ks.append(B.flatten(k))
        ks = B.to_numpy(B.stack(*ks, axis=0))

        # Normalise predicted kernel.
        var_mean = B.mean(ks[:, 0])
        ks = ks/var_mean
        wbml.out.kv('Mean variance of kernel prediction', var_mean)

        k_mean = B.mean(ks, axis=0)
        k_68 = (np.percentile(ks, 32, axis=0),
                np.percentile(ks, 100 - 32, axis=0))
        k_95 = (np.percentile(ks, 2.5, axis=0),
                np.percentile(ks, 100 - 2.5, axis=0))
        k_99 = (np.percentile(ks, 0.15, axis=0),
                np.percentile(ks, 100 - 0.15, axis=0))
        return collect(t=t_k,
                       mean=k_mean,
                       err_68=k_68,
                       err_95=k_95,
                       err_99=k_99,
                       samples=B.transpose(ks)[:, :3])

    def predict_fourier(self):
        """Predict Fourier features.

        Returns:
            tuple: Marginals of the predictions.
        """
        mu_z = B.trisolve(B.transpose(self.L_inv_cov_z), self.root,
                          lower_a=False)
        cov_z = B.cholsolve(self.L_inv_cov_z, B.eye(self.L_inv_cov_z))
        q_z = Normal(cov_z, mu_z)
        return [B.to_numpy(x) for x in q_z.marginals()]

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
