import lab as B
from stheno import Normal
import numpy as np

from .gprv.integrals import ihx, I_ux, I_hz, I_uz
from .gprv.kernel_approx import K_z, K_z_inv, K_u, kernel_approx_u
from .util import collect

__all__ = ['construct', 'predict', 'predict_kernel', 'predict_fourier']


def construct(model, t, y, sigma, mu_u, cov_u):
    """Construct quantities to perform compute the ELBO, perform prediction, et
    cetera.

    Args:
        model (object): Model object.
        t (vector): Time points of data.
        y (vector): Data.
        sigma (scalar): Standard deviation of observation noise.
        mu_u (matrix): Mean of approximate posterior.
        cov_u (matrix): Covariance of approximate posterior.

    Returns:
        :class:`collections.namedtuple`: Computed quantities.
    """

    n = B.length(t)
    q_u = Normal(cov_u, mu_u)

    I_hx_ = ihx(model, t, t)
    I_ux_ = I_ux(model)  # Does not depend on `t`!
    I_hz_ = I_hz(model, t)
    I_uz_ = I_uz(model, t)

    # Put data dimensions first.
    I_uz_ = B.transpose(I_uz_, perm=(2, 0, 1))
    I_hz_ = B.transpose(I_hz_, perm=(2, 0, 1))

    # Construct kernel matrices.
    K_z_ = K_z(model)
    K_z_inv_ = K_z_inv(model)
    # TODO: Use Woodbury determinant lemma here.
    L_z = B.chol(B.reg(K_z_))
    K_u_ = K_u(model)
    L_u = B.chol(B.reg(K_u_))
    # TODO: Optimise this!
    K_u_inv = B.cholsolve(L_u, B.eye(L_u))

    # Do some precomputations.
    L_u_tiled = B.tile(L_u[None, :, :], n, 1, 1)
    L_u_inv_I_uz_ = B.trisolve(L_u_tiled, I_uz_)
    I_zu_inv_K_u_I_uz = B.mm(L_u_inv_I_uz_, L_u_inv_I_uz_, tr_a=True)
    I_zu_inv_K_u_I_uz_sum = B.sum(I_zu_inv_K_u_I_uz, axis=0)

    I_hx_sum = B.sum(I_hx_, axis=0)
    I_ux_sum = n*I_ux_  # Independent of `t`!
    I_hz_sum = B.sum(I_hz_, axis=0)
    I_uz_sum = B.sum(y[:, None, None]*I_uz_, axis=0)  # Weighted by data.

    A_sum = I_ux_sum - \
            B.sum(B.mm(B.mm(I_uz_, K_z_inv_), I_uz_, tr_b=True), axis=0)
    B_sum = I_hz_sum - I_zu_inv_K_u_I_uz_sum
    c_sum = I_hx_sum - \
            B.sum(K_u_inv*I_ux_sum) - \
            B.sum(K_z_inv_*I_hz_sum) + \
            B.sum(K_z_inv_*I_zu_inv_K_u_I_uz_sum)

    # Compute optimal q(z).
    inv_cov_z = K_z_ + 1/sigma**2* \
                B.sum(B_sum + B.mm(B.mm(I_uz_, B.dense(q_u.m2), tr_a=True),
                                   I_uz_),
                      axis=0)
    inv_cov_z_mu_z = 1/sigma**2*B.mm(I_uz_sum, mu_u, tr_a=True)
    L_inv_cov_z = B.chol(B.reg(inv_cov_z))
    root = B.trisolve(L_inv_cov_z, inv_cov_z_mu_z)

    # TODO: Optimise this!
    p_u = Normal(K_u_inv)

    return collect(model=model,

                   n=n,
                   t=t,
                   y=y,
                   sigma=sigma,

                   L_z=L_z,
                   K_z_inv_=K_z_inv_,

                   K_u_=K_u_,
                   K_u_inv=K_u_inv,

                   I_hx_=I_hx_,
                   I_ux_=I_ux_,
                   I_hz_=I_hz_,
                   I_uz_=I_uz_,

                   I_zu_inv_K_u_I_uz=I_zu_inv_K_u_I_uz,

                   A_sum=A_sum,
                   B_sum=B_sum,
                   c_sum=c_sum,

                   L_inv_cov_z=L_inv_cov_z,
                   root=root,

                   p_u=p_u,
                   q_u=q_u)


def elbo(c):
    """Compute the ELBO.

    Args:
        c (:class:`collections.namedtuple`): Computed quantities.

    Returns:
        scalar: ELBO.
    """
    return -0.5*c.n*B.log(2*B.pi*c.sigma**2) + \
           -0.5/c.sigma**2*B.sum(c.y**2) + \
           -0.5/c.sigma**2*B.sum(c.q_u.mean*B.mm(c.A_sum, c.q_u.mean)) + \
           -0.5/c.sigma**2*B.sum(c.q_u.var*c.A_sum) + \
           -0.5/c.sigma**2*c.c_sum + \
           -0.5*2*B.sum(B.log(B.diag(c.L_z))) + \
           0.5*2*B.sum(B.log(B.diag(c.L_inv_cov_z))) + \
           0.5*B.sum(c.root**2) - \
           c.q_u.kl(c.p_u)


def predict(c):
    """Predict at the data.

    Args:
        c (:class:`collections.namedtuple`): Computed quantities.

    Returns:
        tuple: Tuple containing the mean and standard deviation of the
            predictions.
    """

    # Construct optimal q(z).
    mu_z = B.trisolve(B.transpose(c.L_inv_cov_z), c.root, lower_a=False)
    cov_z = B.cholsolve(c.L_inv_cov_z, B.eye(c.L_inv_cov_z))
    q_z = Normal(cov_z, mu_z)

    # Compute mean.
    mu = B.flatten(B.mm(B.mm(c.q_u.mean, c.I_uz_, tr_a=True), mu_z))

    # Compute variance.
    A = c.I_ux_ - B.mm(B.mm(c.I_uz_, c.K_z_inv_), c.I_uz_, tr_b=True)
    B_ = c.I_hz_ - c.I_zu_inv_K_u_I_uz
    c_ = c.I_hx_ - \
         B.sum(c.K_u_inv*c.I_ux_) - \
         B.sum(B.sum(c.K_z_inv_[None, :, :]*c.I_hz_, axis=1), axis=1) + \
         B.sum(B.sum(c.K_z_inv_[None, :, :]*c.I_zu_inv_K_u_I_uz, axis=1),
               axis=1)
    var = B.sum(B.sum(A*c.q_u.m2[None, :, :], axis=1), axis=1) + \
          B.sum(B.sum(B_*q_z.m2[None, :, :], axis=1), axis=1) + c_

    return B.to_numpy(mu), B.to_numpy(B.sqrt(var))


def predict_kernel(c):
    """Predict kernel.

    Args:
        c (:class:`collections.namedtuple`): Computed quantities.

    Returns:
        :class:`collections.namedtuple`: Named tuple containing the prediction.
    """
    # Transform back to normal space.
    q_u_orig = c.q_u.lmatmul(c.K_u_)

    ks = []
    t_k = B.linspace(c.model.dtype, 0, 1.2*B.max(c.model.t_u), 100)
    for i in range(100):
        k = kernel_approx_u(c.model,
                            t1=t_k,
                            t2=B.zeros(c.model.dtype, 1),
                            u=B.flatten(q_u_orig.sample()))
        ks.append(B.flatten(k))

    ks = B.to_numpy(B.stack(*ks, axis=0))
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


def predict_fourier(c):
    """Predict Fourier features.

    Args:
        c (:class:`collections.namedtuple`): Computed quantities.

    Returns:
        tuple: Marginals of the predictions.
    """
    mu_z = B.trisolve(B.transpose(c.L_inv_cov_z), c.root, lower_a=False)
    cov_z = B.cholsolve(c.L_inv_cov_z, B.eye(c.L_inv_cov_z))
    q_z = Normal(cov_z, mu_z)
    return [B.to_numpy(x) for x in q_z.marginals()]
