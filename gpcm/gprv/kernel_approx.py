from lab import B

from .integrals import ihx, iux

__all__ = ['kernel_approx_u',
           'ku',
           'K_z',
           'K_u',
           'K_u_inv',
           'K_z_inv',
           'log_det_K_z']


def K_u(model):
    """Covariance matrix :math:`K_u` of :math:`u(t)` at inputs `t_u_vec`.

    Args:
        model (:class:`.model.GPRV`): Model.

    Returns:
        matrix: :math:`K_u` of shape `(len(model.t_u), len(model.t_u))`.
    """

    t_u = B.flatten(model.t_u)
    return ku(model, t_u[None, :], t_u[:, None])


def K_u_inv(model, t_u, is_toeplitz=False):
    """Inverse :math:`K_u^{-1}` of :math:`K_u` for :math:`u(t)` at inputs
    `t_u_vec`.

    Args:
        model (:class:`.model.GPRV`): Model.
        t_u (vector): Inputs of inducing variables, which will be flattened.
        is_toeplitz (bool, optional): `t_u_vec` are equally spaced.

    Returns:
        matrix: :math:`K_u^{-1}` of shape `(len(t_u_vec), len(t_u_vec))`.
    """
    if is_toeplitz:
        t_u = B.flatten(t_u)
        col = ku(model, t_u, t_u[0])
        return B.toepsolve(col, B.eye(B.length(col)))
    else:
        chol = B.reg(B.chol(K_u(model)))
        return B.cholsolve(chol, B.eye(chol))


def ku(model, t_u_1, t_u_2):
    """Covariance function of inducing variables :math:`u` associated with
    :math:`h`.

    Args:
        model (:class:`.model.GPRV`): Model.
        t_u_1 (tensor): First inducing point location input.
        t_u_2 (tensor): Second inducing point location input.

    Returns:
        tensor: Kernel matrix broadcasted over `tu1` and `tu2`.
    """
    return model.gamma_t**2/model.gamma/2*B.exp(-model.gamma*B.abs(t_u_1 - t_u_2))


def kernel_approx_u(model, t1, t2, u):
    """Kernel approximation using inducing variables :math:`u` for the impulse
    response :math:`h`.

    Args:
        model (:class:`.model.GPRV`): Model.
        t1 (tensor): First time input.
        t2 (tensor): Second time input.
        u (tensor): Values of the inducing variables.

    Returns:
        tensor: Approximation of the kernel matrix broadcasted over `t1` and
            `t2`.
    """
    # Construct the first part.
    part1 = ihx(model, t1[:, None], t2[None, :])

    # Construct the second part.
    Lu = B.cholesky(K_u(model))
    iLu = B.trisolve(Lu, B.eye(Lu))
    iLu_u = B.mm(iLu, u[:, None])
    Kiux = iux(model,
               # Increase ranks for broadcasting.
               t1[:, None, None, None],
               t2[None, :, None, None],
               model.t_u[None, None, :, None],
               model.t_u[None, None, None, :])
    trisolved = B.mm(B.mm(iLu, Kiux, tr_a=True), iLu)
    part2 = B.trace(trisolved, axis1=2, axis2=3) - \
            B.trace(B.mm(B.mm(iLu_u, trisolved, tr_a=True), iLu_u),
                    axis1=2, axis2=3)

    return part1 - part2


def log_det_K_z(model):
    """Computes :math:`\log \det (K_z)` of :math:`K_z` for :math:`z_m` with
    :math:`m=0,...,2M`.

    Args:
        model (:class:`.model.GPRV`): Model.

    Returns:
        scalar: :math:`\log \det (K_z)`.
    """
    alpha, beta = K_z(model, output_alpha_beta=True)
    return B.log(1 + B.sum(beta**2/alpha)) + B.sum(B.log(alpha))


def K_z_inv(model):
    """Inverse :math:`K_z^{-1}` of :math:`K_z` for :math:`z_m` for
    :math:`m=0,\ldots,2M`.

    Args:
        model (:class:`.model.GPRV`): Model.

    Returns:
        matrix: :math:`K_z^{-1}`.
    """
    alpha, beta = K_z(model, output_alpha_beta=True)
    denom = 1 + B.sum(beta**2/alpha)
    div = beta/alpha
    return B.diag(1/alpha) - div[:, None]*div[None, :]/denom


def K_z(model, output_alpha_beta=False):
    """Covariance matrix :math:`K_z` of :math:`z_m` for :math:`m=0,\ldots,2M`.

    Args:
        model (:class:`.model.GPRV`): Model.
        output_alpha_beta (bool, optional): Return `alpha` and `beta` instead.
            Defaults to `False`.

    Returns:
        matrix: :math:`K_z`.
    """

    # Compute harmonic frequencies.
    m = model.ms - B.cast(model.dtype, (model.ms > model.m_max))*model.m_max
    omega = 2*B.pi*m/(model.b - model.a)

    # Compute the vectors alpha and beta for K.
    lam_t = 1
    psd = psd_matern_05(omega, model.lam, lam_t)
    alpha_tmp = (model.b - model.a)/2*psd**(-1)
    alpha = alpha_tmp + alpha_tmp*B.cast(B.dtype(alpha_tmp), model.ms == 0)
    beta = 1/(lam_t**.5)*B.cast(model.dtype, model.ms <= model.m_max)

    # Return the right thing.
    if output_alpha_beta:
        return alpha, beta
    else:
        return B.diag(alpha) + beta[:, None]*beta[None, :]


def psd_matern_05(omega, lam, lam_t):
    """Spectral density of Matern-1/2 process.

    Args:
        omega (tensor): Frequency.
        lam (tensor): Decay.
        lam_t (tensor): Scale.

    Returns:
        tensor: Spectral density.
    """
    return 2*lam_t*lam/(lam**2 + omega**2)
