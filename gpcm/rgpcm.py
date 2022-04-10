import warnings

import lab as B
import numpy as np
from matrix import Dense, Diagonal, LowRank
from mlkernels import Exp

from .model import AbstractGPCM
from .util import invert_perm, method

__all__ = ["RGPCM"]


class RGPCM(AbstractGPCM):
    """Rough Gaussian Process Convolution Model.

    Args:
        scheme (str, optional): Approximation scheme. Defaults to `structured`.
        noise (scalar, optional): Observation noise. Defaults to `1e-4`.
        fix_noise (bool, optional): Do not learn the observation noise. Defaults
            to `False`.
        alpha (scalar, optional): Decay of the window.
        alpha_t (scalar, optional): Scale of the window. Defaults to normalise the
            window to unity power.
        window (scalar, alternative): Length of the window. This will be used to
            determine `alpha` if it is not given.
        fix_window (bool, optional): Do not learn the window. Defaults to `False`.
        lam (scalar, optional): Decay of the kernel of :math:`x`. Defaults to
            `1 / scale`.
        gamma (scalar, optional): Decay of the transform :math:`u` of :math:`x`.
            Defaults to the inverse of ten times the spacing between locations of
            data points.
        gamma_t (scalar, optional): Scale of the transform. Defaults to normalise the
            transform to unity power.
        a (scalar, optional): Lower bound of support of the basis.
        b (scalar, optional): Upper bound of support of the basis.
        m_max (int, optional): Defines cosine and sine basis functions.
        m_max_cap (int, optional): Maximum value for `m_max`. Defaults to `150`.
        n_z (int, optional): Number of inducing points for :math:`z`. Will be used
            to determine `m_max`.
        scale (scalar, alternative): Length scale of the function. This will be used
            to initialise `lam` and determine `m_max` if `m_max` is not given.
        fix_scale (bool, optional): Do not learn the length scale. Defaults to `False`.
        ms (vector, optional): Basis function frequencies. Defaults to
            :math:`0,\\ldots,2M-1`.
        n_u (int, optional): Number of inducing points for :math:`u`.
        n_u_cap (int, optional): Maximum value for `n_u`. Defaults to `300`.
        t_u (vector, optional): Locations of inducing points for :math:`u`. Defaults
            to equally spaced points across twice the filter length scale.
        extend_t_z (bool, optional): Does nothing. Provided to ensure a consistent
            interface between all models.
        t (vector, alternative): Locations of interest. Can be used to automatically
            initialise quantities.
    """

    name = "RGPCM"
    """str: Formatted name."""

    def __init__(
        self,
        scheme="structured",
        noise=5e-2,
        fix_noise=False,
        alpha=None,
        alpha_t=None,
        window=None,
        fix_window=False,
        lam=None,
        gamma=None,
        gamma_t=None,
        a=None,
        b=None,
        m_max=None,
        m_max_cap=150,
        n_z=None,
        scale=None,
        fix_scale=False,
        ms=None,
        n_u=None,
        n_u_cap=300,
        t_u=None,
        extend_t_z=None,
        t=None,
    ):
        AbstractGPCM.__init__(self, scheme)

        # Ensure that `t` is a vector.
        if t is not None:
            t = np.array(t)

        # Store whether to fix the length scale, window length, and noise.
        self.fix_scale = fix_scale
        self.fix_window = fix_window
        self.fix_noise = fix_noise

        # First initialise optimisable model parameters.
        if alpha is None:
            alpha = 1 / window

        if alpha_t is None:
            alpha_t = B.sqrt(2 * alpha)

        if lam is None:
            lam = 1 / scale

        self.noise = noise
        self.alpha = alpha
        self.alpha_t = alpha_t
        self.lam = lam

        # For convenience, also store the extent of the filter.
        self.extent = 4 / self.alpha

        # Then initialise fixed variables.
        if t_u is None:
            # Place inducing points until the filter is `exp(-pi) = 4.32%`.
            t_u_max = B.pi / self.alpha

            # `n_u` is required to initialise `t_u`.
            if n_u is None:
                # Set it to two inducing points per wiggle, multiplied by two to account
                # for the longer range.
                n_u = int(np.ceil(2 * 2 * window / scale))
                if n_u > n_u_cap:
                    warnings.warn(
                        f"Using {n_u} inducing points for the filter, which is too "
                        f"many. It is capped to {n_u_cap}.",
                        category=UserWarning,
                    )
                    n_u = n_u_cap

            t_u = B.linspace(0, t_u_max, n_u)

        if n_u is None:
            n_u = B.shape(t_u)[0]

        if a is None:
            a = B.min(t) - B.max(t_u)

        if b is None:
            b = B.max(t)

        # First, try to determine `m_max` from `n_z`.
        if m_max is None and n_z is not None:
            m_max = int(np.ceil(n_z / 2))

        if m_max is None:
            freq = 1 / scale
            m_max = int(np.ceil(freq * (b - a)))
            if m_max > m_max_cap:
                warnings.warn(
                    f"Using {m_max} inducing features, which is too "
                    f"many. It is capped to {m_max_cap}.",
                    category=UserWarning,
                )
                m_max = m_max_cap

        if ms is None:
            ms = B.range(2 * m_max + 1)

        self.a = a
        self.b = b
        self.m_max = m_max
        self.ms = ms
        self.n_z = len(ms)
        self.n_u = n_u
        self.t_u = t_u

        # Initialise dependent model parameters.
        if gamma is None:
            gamma = 1 / (2 * (self.t_u[1] - self.t_u[0]))

        if gamma_t is None:
            gamma_t = B.sqrt(2 * gamma)

        # Must ensure that `gamma < alpha`.
        self.gamma = min(gamma, self.alpha / 1.5)
        self.gamma_t = gamma_t

    def k_h(self):
        """Get the kernel function of the filter.

        Returns:
            :class:`mlkernels.Kernel`: Kernel for :math:`h`.
        """
        k_h = Exp().stretch(1 / self.lam)  # Kernel of filter before window
        k_h *= lambda t: B.exp(-self.alpha * B.abs(t))  # Window
        k_h *= lambda t: B.cast(self.dtype, t >= 0)  # Causality constraint
        return k_h

    def __prior__(self):
        gamma_factor_init = self.alpha / self.gamma

        # Make parameters learnable:
        if not self.fix_noise:
            self.noise = self.ps.positive(self.noise, name="noise")
        if not self.fix_window:
            self.alpha = self.ps.positive(self.alpha, name="alpha")
        self.alpha_t = self.ps.positive(self.alpha_t, name="alpha_t")
        if not self.fix_scale:
            self.lam = self.ps.positive(self.lam, name="lambda")
        # We must ensure that `gamma < alpha`.
        gamma_factor = self.ps.bounded(
            gamma_factor_init,
            lower=1,
            upper=1e4,
            name="gamma_factor",
        )
        self.gamma = self.alpha / gamma_factor
        self.gamma_t = self.gamma_t  # Do not learn variable of inter-domain transform!
        # Bound the inducing points so they don't move away too far.
        self.t_u = self.ps.bounded(
            self.t_u,
            lower=-1e-4,  # First inducing point is at the origin.
            upper=max(self.t_u) * 1.5,
            name="t_u",
        )

        AbstractGPCM.__prior__(self)


@method(RGPCM)
def compute_K_u(model):
    """Covariance matrix of inducing variables :math:`u` associated with
    :math:`h`.

    Args:
        model (:class:`.gprv.GPRV`): Model.

    Returns:
        tensor: :math:`K_u`.
    """
    return Dense(
        (model.gamma_t**2 / (2 * model.gamma))
        * B.exp(-model.gamma * B.abs(model.t_u[:, None] - model.t_u[None, :]))
    )


def psd_matern_12(omega, lam, lam_t):
    """Spectral density of Matern-1/2 process.

    Args:
        omega (tensor): Frequency.
        lam (tensor): Decay.
        lam_t (tensor): Scale.

    Returns:
        tensor: Spectral density.
    """
    return 2 * lam_t * lam / (lam**2 + omega**2)


@method(RGPCM)
def compute_K_z(model):
    """Covariance matrix :math:`K_z` of :math:`z_m` for :math:`m=0,\\ldots,2M`.

    Args:
        model (:class:`.gprv.GPRV`): Model.

    Returns:
        matrix: :math:`K_z`.
    """
    # Compute harmonic frequencies.
    m = model.ms - B.cast(model.dtype, model.ms > model.m_max) * model.m_max
    omega = 2 * B.pi * m / (model.b - model.a)

    # Compute the parameters of the kernel matrix.
    lam_t = 1
    alpha = 0.5 * (model.b - model.a) / psd_matern_12(omega, model.lam, lam_t)
    alpha = alpha + alpha * B.cast(model.dtype, model.ms == 0)
    beta = 1 / (lam_t**0.5) * B.cast(model.dtype, model.ms <= model.m_max)

    return Diagonal(alpha) + LowRank(left=beta[:, None])


@method(RGPCM)
def compute_i_hx(model, t1=None, t2=None):
    """Compute the :math:`I_{hx}` integral.

    Args:
        model (:class:`.gprv.GPRV`): Model.
        t1 (tensor, optional): First time input. Defaults to zero.
        t2 (tensor, optional): Second time input. Defaults to zero.

    Returns:
        tensor: Value of :math:`I_{hx}` for all `t1` and `t2`.
    """
    if t1 is None:
        t1 = B.zero(model.dtype)
    if t2 is None:
        t2 = B.zero(model.dtype)
    return model.alpha_t**2 / 2 / model.alpha * B.exp(-model.lam * B.abs(t1 - t2))


@method(RGPCM)
def compute_I_ux(model, t1=None, t2=None):
    """Compute the :math:`I_{ux}` integral.

    Args:
        model (:class:`.gprv.GPRV`): Model.
        t1 (tensor, optional): First time input. Defaults to zero.
        t2 (tensor, optional): Second time input. Defaults to zero.

    Returns:
        tensor: Value of :math:`I_{ux}` for all `t1`, `t2`.
    """
    if t1 is None:
        t1 = B.zeros(model.dtype, 1)
        squeeze_t1 = True
    else:
        squeeze_t1 = False

    if t2 is None:
        t2 = B.zeros(model.dtype, 1)
        squeeze_t2 = True
    else:
        squeeze_t2 = False

    t1 = t1[:, None, None, None]
    t2 = t2[None, :, None, None]
    t_u_1 = model.t_u[None, None, :, None]
    t_u_2 = model.t_u[None, None, None, :]
    ga = model.gamma - model.alpha
    result = (
        model.alpha_t**2
        * model.gamma_t**2
        * B.exp(-model.gamma * (t_u_1 + t_u_2) + ga * (t1 + t2))
        * integral_abcd_lu(-t1, t_u_2 - t1, -t2, t_u_1 - t2, ga, model.lam)
    )

    if squeeze_t1 and squeeze_t2:
        return result[0, 0, :, :]
    elif squeeze_t1:
        return result[0, :, :, :]
    elif squeeze_t2:
        return result[:, 0, :, :]
    else:
        return result


def integral_abcd(a, b, c, d):
    """Compute the a-b-c-d integral from the paper.

    Args:
        a (tensor): First upper integration bound.
        b (tensor): Second upper integration bound.
        c (tensor): Decay for sum.
        d (tensor): Decay for absolute difference.

    Returns:
        tensor: Value of the integral.
    """
    # Compute the conditional and signs.
    sign = B.sign(a)
    condition = a * b >= 0

    # Compute the two parts.
    part1 = sign * d / c * (1 - B.exp(2 * c * sign * B.minimum(B.abs(a), B.abs(b))))
    part2 = (
        1
        - B.exp(c * a - d * B.abs(a))
        - B.exp(c * b - d * B.abs(b))
        + B.exp(c * (a + b) - d * B.abs(a - b))
    )

    # Combine and return.
    condition = B.cast(B.dtype(part1), condition)
    return (condition * part1 + part2) / (c**2 - d**2)


def integral_abcd_lu(a_lb, a_ub, b_lb, b_ub, c, d):
    """Compute the a-b-c-d integral with lower and upper bounds from the paper.

    Args:
        a_lb (tensor): First lower integration bound.
        a_ub (tensor): First upper integration bound.
        b_lb (tensor): Second lower integration bound.
        b_ub (tensor): Second upper integration bound.
        c (tensor): Decay for sum.
        d (tensor): Decay for absolute difference.

    Returns:
        tensor: Value of the integral.
    """
    return (
        integral_abcd(a_ub, b_ub, c, d)
        + integral_abcd(a_lb, b_lb, c, d)
        + -integral_abcd(a_ub, b_lb, c, d)
        + -integral_abcd(a_lb, b_ub, c, d)
    )


@method(RGPCM)
def compute_I_hz(model, t):
    """Compute the :math:`I_{hz,t_i}` matrix for :math:`t_i` in `t`.

    Args:
        model (:class:`.gprv.GPRV`): Model.
        t (vector): Time points of data.

    Returns:
        tensor: Value of :math:`I_{hz,t_i}`, with shape
            `(len(t), len(model.ms), len(model.ms))`.
    """
    # Compute sorting permutation.
    perm = np.argsort(model.ms)
    inverse_perm = invert_perm(perm)

    # Sort to allow for simple concatenation.
    m_max = model.m_max
    ms = model.ms[perm]

    # Construct I_hz for m,n <= M.
    ns = ms[ms <= m_max]
    I_0_cos_1 = _I_hx_0_cos(
        model, -ns[None, :, None] + ns[None, None, :], t[:, None, None]
    )
    I_0_cos_2 = _I_hx_0_cos(
        model, ns[None, :, None] + ns[None, None, :], t[:, None, None]
    )
    I_hz_mnleM = 0.5 * (I_0_cos_1 + I_0_cos_2)

    # Construct I_hz for m,n > M.
    ns = ms[ms > m_max] - m_max
    I_0_cos_1 = _I_hx_0_cos(
        model, -ns[None, :, None] + ns[None, None, :], t[:, None, None]
    )
    I_0_cos_2 = _I_hx_0_cos(
        model, ns[None, :, None] + ns[None, None, :], t[:, None, None]
    )
    I_hz_mngtM = 0.5 * (I_0_cos_1 - I_0_cos_2)

    # Construct I_hz for 0 < m <= M and n > M.
    ns = ms[(0 < ms) * (ms <= m_max)]
    ns2 = ms[ms > m_max]  # Do not subtract M!
    I_0_sin_1 = _I_hx_0_sin(
        model, ns[None, :, None] + ns2[None, None, :], t[:, None, None]
    )
    I_0_sin_2 = _I_hx_0_sin(
        model, -ns[None, :, None] + ns2[None, None, :], t[:, None, None]
    )
    I_hz_mleM_ngtM = 0.5 * (I_0_sin_1 + I_0_sin_2)

    # Construct I_hz for m = 0 and n > M.
    ns = ms[ms == 0]
    ns2 = ms[ms > m_max]  # Do not subtract M!
    I_hz_0_gtM = _I_hx_0_sin(
        model, ns[None, :, None] + ns2[None, None, :], t[:, None, None]
    )

    # Concatenate to form I_hz for m <= M and n > M.
    I_hz_mleM_ngtM = B.concat(I_hz_0_gtM, I_hz_mleM_ngtM, axis=1)

    # Compute the other half by transposing.
    I_hz_mgtM_nleM = B.transpose(I_hz_mleM_ngtM, perm=(0, 2, 1))

    # Construct result.
    result = B.concat(
        B.concat(I_hz_mnleM, I_hz_mleM_ngtM, axis=2),
        B.concat(I_hz_mgtM_nleM, I_hz_mngtM, axis=2),
        axis=1,
    )

    # Undo sorting.
    result = B.take(result, inverse_perm, axis=1)
    result = B.take(result, inverse_perm, axis=2)

    return result


def _I_hx_0_cos(model, n, t):
    """Compute the :math:`I_{0,n:\\cos}` integral."""
    omega_n = 2 * B.pi * n / (model.b - model.a)
    t_less_a = t - model.a
    return (
        model.alpha_t**2
        / (4 * model.alpha**2 + omega_n**2)
        * (
            (
                2
                * model.alpha
                * (B.cos(omega_n * t_less_a) - B.exp(-2 * model.alpha * t_less_a))
            )
            + omega_n * B.sin(omega_n * t_less_a)
        )
    ) + (
        model.alpha_t**2
        / (2 * (model.alpha + model.lam))
        * B.exp(-2 * model.alpha * t_less_a)
    )


def _I_hx_0_sin(model, n, t):
    """Compute the :math:`I_{0,n:\\sin}`, :math:`n>M`, integral."""
    omega = 2 * B.pi * (n - model.m_max) / (model.b - model.a)
    t_less_a = t - model.a
    return (
        model.alpha_t**2
        / (4 * model.alpha**2 + omega**2)
        * (
            omega * (B.exp(-2 * model.alpha * t_less_a) - B.cos(omega * t_less_a))
            + 2 * model.alpha * B.sin(omega * t_less_a)
        )
    )


@method(RGPCM)
def compute_I_uz(model, t):
    """Compute the :math:`I_{uz,t_i}` matrix for :math:`t_i` in `t`.

    Args:
        model (:class:`.gprv.GPRV`): Model.
        t (vector): Time points :math:`t_i` of data.

    Returns:
        tensor: Value of :math:`I_{uz,t_i}`, with shape
            `(len(t), len(model.t_u), len(model.ms))`.
    """
    # Compute sorting permutation.
    perm = np.argsort(model.ms)
    inverse_perm = invert_perm(perm)

    # Sort to allow for simple concatenation.
    ms = model.ms[perm]

    # Part of the factor is absorbed in the integrals.
    factor = (
        model.alpha_t * model.gamma_t * B.exp(-model.gamma * model.t_u[None, :, None])
    )

    # Compute I(l, u, 0).
    k = ms[ms == 0]
    I_uz0 = (
        _integral_lu0(
            model, t[:, None, None] - model.t_u[None, :, None], t[:, None, None]
        )
        + k[None, None, :]
    )  # This is all zeros, but performs broadcasting.

    # Compute I(l, u, k) for 0 < k < M + 1.
    k = ms[(0 < ms) * (ms <= model.m_max)]
    I_uz_leM = _integral_luk_leq_M(
        model,
        t[:, None, None] - model.t_u[None, :, None],
        t[:, None, None],
        k[None, None, :],
    )

    # Compute I(l, u, k) for k > M.
    k = ms[ms > model.m_max]
    I_uz_gtM = _integral_luk_g_M(
        model,
        t[:, None, None] - model.t_u[None, :, None],
        t[:, None, None],
        k[None, None, :],
    )

    # Construct result.
    result = B.concat(I_uz0, I_uz_leM, I_uz_gtM, axis=2)

    # Undo sorting and return.
    return factor * B.take(result, inverse_perm, axis=2)


def _integral_lu0(model, l, u):
    """Compute :math:`I(l,u,k)` for :math:`k=0`."""
    ag = model.alpha - model.gamma
    return 1 / ag * (1 - B.exp(ag * (l - u)))


def _integral_luk_leq_M(model, l, u, k):
    """Compute :math:`I(l,u,k)` for :math:`0<k<M+1`. Assumes that :math:`a<l,u<b`."""
    ag = model.alpha - model.gamma
    om = 2 * B.pi * k / (model.b - model.a)
    return (1 / (ag**2 + om**2)) * (
        (ag * B.cos(om * (u - model.a)) + om * B.sin(om * (u - model.a)))
        - (
            B.exp(ag * (l - u))
            * (ag * B.cos(om * (l - model.a)) + om * B.sin(om * (l - model.a)))
        )
    )


def _integral_luk_g_M(model, l, u, k):
    """Internal function :math:`I(l,u,k)` for :math:`k>M`. Assumes that :math:`a<l,u<b`.

    Note:
        This function gives :math:`-I(l,u,k)`, i.e. *minus* the integral in the paper!
    """
    ag = model.alpha - model.gamma
    om = 2 * B.pi * (k - model.m_max) / (model.b - model.a)
    return (-1 / (ag**2 + om**2)) * (
        (-ag * B.sin(om * (u - model.a)) + om * B.cos(om * (u - model.a)))
        - (
            B.exp(ag * (l - u))
            * (-ag * B.sin(om * (l - model.a)) + om * B.cos(om * (l - model.a)))
        )
    )
