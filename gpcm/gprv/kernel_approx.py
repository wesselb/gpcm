import lab as B
from matrix import Diagonal, LowRank, Dense

__all__ = ['k_u', 'K_z', 'K_u']


def k_u(model, t_u_1, t_u_2):
    """Covariance function of inducing variables :math:`u` associated with
    :math:`h`.

    Args:
        model (:class:`.model.GPRV`): Model.
        t_u_1 (tensor): First inducing point location input.
        t_u_2 (tensor): Second inducing point location input.

    Returns:
        tensor: Kernel matrix broadcasted over `tu1` and `tu2`.
    """
    return (model.gamma_t**2/model.gamma/2*
            B.exp(-model.gamma*B.abs(t_u_1 - t_u_2)))


def K_u(model):
    """Covariance matrix :math:`K_u` of :math:`u(t)` at inputs `t_u_vec`.

    Args:
        model (:class:`.model.GPRV`): Model.

    Returns:
        matrix: :math:`K_u` of shape `(len(model.t_u), len(model.t_u))`.
    """

    t_u = B.flatten(model.t_u)
    return Dense(k_u(model, t_u[None, :], t_u[:, None]))


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


def K_z(model):
    """Covariance matrix :math:`K_z` of :math:`z_m` for :math:`m=0,\ldots,2M`.

    Args:
        model (:class:`.model.GPRV`): Model.

    Returns:
        matrix: :math:`K_z`.
    """
    # Compute harmonic frequencies.
    m = model.ms - B.cast(model.dtype, (model.ms > model.m_max))*model.m_max
    omega = 2*B.pi*m/(model.b - model.a)

    # Compute the parameters of the kernel matrix.
    lam_t = 1
    psd = psd_matern_05(omega, model.lam, lam_t)
    alpha_tmp = (model.b - model.a)/2*psd**(-1)
    alpha = alpha_tmp + alpha_tmp*B.cast(B.dtype(alpha_tmp), model.ms == 0)
    beta = 1/(lam_t**.5)*B.cast(model.dtype, model.ms <= model.m_max)

    return Diagonal(alpha) + LowRank(left=beta[:, None])
