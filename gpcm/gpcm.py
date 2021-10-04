import warnings

import lab as B
import numpy as np
from matrix import Dense
from varz import Vars

from .exppoly import ExpPoly, const, var
from .model import AbstractGPCM
from .util import method

__all__ = ["GPCM", "CGPCM"]


def scale_to_factor(scale):
    """Convert a length scale to a factor.

    Args:
        scale (tensor): Length scale to convert to a factor.

    Returns:
        tensor: Equivalent factor.
    """
    return (B.pi / 2) / (2 * scale ** 2)


def factor_to_scale(factor):
    """Convert a factor to a length scale.

    Args:
        factor (tensor): Factor to convert to a length scale.

    Returns:
        tensor: Equivalent length scale.
    """
    return 1 / B.sqrt(4 * factor / B.pi)


class GPCM(AbstractGPCM):
    """GPCM and causal variant.

    Args:
        scheme (str, optional): Approximation scheme. Defaults to `structured`.
        causal (bool, optional): Use causal variant. Defaults to `False`.
        alpha (scalar, optional): Decay of the window.
        alpha_t (scalar, optional): Scale of the window. Defaults to
            normalise the window to unity power.
        window (scalar, alternative): Length of the window. This will be used
            to determine `alpha` if it is not given.
        gamma (scalar, optional): Decay on the prior of :math:`h`.
        scale (scalar, alternative): Length scale of the function. This will be
            used to determine `gamma` if it is not given.
        omega (scalar, optional): Decay of the transform :math:`s` of
            :math:`x`. Defaults to length scale half the spacing
            between the inducing points.
        n_u (int, optional): Number of inducing points for :math:`u`.
        n_u_cap (int, optional): Maximum value for `n_u`. Defaults to `150`.
        t_u (vector, optional): Locations of inducing points for :math:`u`.
            Defaults to equally spaced points across twice the filter length
            scale.
        n_z (int, optional): Number of inducing points for :math:`s`.
        n_z_cap (int, optional): Maximum number of inducing points. Defaults
            to `150`.
        t_z (vector, optional): Locations of inducing points for :math:`s`.
            Defaults to equally spaced points across the span of the data
            extended by twice the filter length scale.
        extend_t_z (bool, optional): Increase the number of inducing points for
            :math:`z` to additionally and appropriately cover an extension of the length
            of the filter on both sides.
        t (vector, alternative): Locations of interest. Can be used to automatically
            initialise quantities.
    """

    name = "GPCM"
    """str: Formatted name."""

    def __init__(
        self,
        scheme="structured",
        causal=False,
        noise=1e-4,
        alpha=None,
        alpha_t=None,
        window=None,
        gamma=None,
        scale=None,
        omega=None,
        n_u=None,
        n_u_cap=150,
        t_u=None,
        n_z=None,
        n_z_cap=150,
        t_z=None,
        extend_t_z=False,
        t=None,
    ):
        AbstractGPCM.__init__(self, scheme)

        # Store whether this is the CGPCM instead of the GPCM.
        self.causal = causal

        # Ensure that `t` is a vector.
        if t is not None:
            t = np.array(t)

        # First initialise optimisable model parameters.
        if alpha is None:
            if causal:
                alpha = scale_to_factor(2 * window)
            else:
                alpha = scale_to_factor(window)

        if alpha_t is None:
            if causal:
                alpha_t = (8 * alpha / B.pi) ** 0.25
            else:
                alpha_t = (2 * alpha / B.pi) ** 0.25

        if gamma is None:
            gamma = scale_to_factor(scale) - 0.5 * alpha

        self.noise = noise
        self.alpha = alpha
        self.alpha_t = alpha_t
        self.gamma = gamma

        # Then initialise fixed variables.
        if t_u is None:
            # For the causal case, decay is less quick, and we want `t_u_max` to be the
            # same in both cases.
            if causal:
                t_u_max = factor_to_scale(self.alpha)
            else:
                t_u_max = 2 * factor_to_scale(self.alpha)

            # `n_u` is required to initialise `t_u`.
            if n_u is None:
                # Set it to four inducing points per wiggle, multiplied by two to account
                # for both sides (acausal model) or the extended filter (causal model).
                n_u = int(np.ceil(4 * 2 * window / scale))
                if n_u > n_u_cap:
                    warnings.warn(
                        f"Using {n_u} inducing points for the filter, which is too "
                        f"many. It is capped to {n_u_cap}.",
                        category=UserWarning,
                    )
                    n_u = n_u_cap

            # For the causal case, only need inducing points on the right side.
            if causal:
                d_t_u = t_u_max / (n_u - 1)
                n_u += 2
                t_u = B.linspace(-2 * d_t_u, t_u_max, n_u)
            else:
                if n_u % 2 == 0:
                    n_u += 1
                t_u = B.linspace(-t_u_max, t_u_max, n_u)

        if n_u is None:
            n_u = B.shape(t_u)[0]

        if t_z is None:
            if n_z is None:
                # Use two inducing points per wiggle.
                n_z = int(np.ceil(2 * (max(t) - min(t)) / scale))
                if n_z > n_z_cap:
                    warnings.warn(
                        f"Using {n_z} inducing points, which is too "
                        f"many. It is capped to {n_z_cap}.",
                        category=UserWarning,
                    )
                    n_z = n_z_cap

            t_z = B.linspace(min(t), max(t), n_z)

        if n_z is None:
            n_z = B.shape(t_z)[0]

        if extend_t_z:
            # Again, decay is less quick for the causal case. See above.
            if causal:
                t_z_extra = factor_to_scale(self.alpha)
            else:
                t_z_extra = 2 * factor_to_scale(self.alpha)

            d_t_u = (max(t) - min(t)) / (n_z - 1)
            n_z_extra = int(np.ceil(t_z_extra / d_t_u))
            t_z_extra = n_z_extra * d_t_u  # Make it align exactly.

            # For the causal case, only need inducing points on the left side.
            if causal:
                n_z += n_z_extra
                t_z = B.linspace(min(t) - t_z_extra, max(t), n_z)
            else:
                n_z += 2 * n_z_extra
                t_z = B.linspace(min(t) - t_z_extra, max(t) + t_z_extra, n_z)

        self.n_u = n_u
        self.t_u = t_u
        self.n_z = n_z
        self.t_z = t_z

        # Initialise dependent optimisable model parameters.
        if omega is None:
            omega = scale_to_factor(0.5 * (self.t_z[1] - self.t_z[0]))

        # Fix variance of inter-domain process to one.
        omega_t = (2 * omega / B.pi) ** 0.25

        self.omega = omega
        self.omega_t = omega_t

    def k_h(self, t1, t2):
        """Kernel function associated to the filter :math:`h`.

        Args:
            t1 (tensor): First input :math:`t_1`.
            t2 (tensor): Second input :math:`t_2`.

        Returns:
            tensor: Kernel tensor :math:`k_h(t_1, t_2)`.
        """
        return ExpPoly(
            self.alpha_t ** 2,
            -(
                const(self.alpha) * (t1 ** 2 + t2 ** 2)
                + const(self.gamma) * (t1 - t2) ** 2
            ),
        )

    def k_xs(self, t1, t2):
        """Covariance function between to the white noise :math:`x` and its interdomain
        transform :math:`s`.

        Args:
            t1 (tensor): First input :math:`t_1`.
            t2 (tensor): Second input :math:`t_2`.

        Returns:
            tensor: Kernel tensor :math:`k_{xs}(t_1, t_2)`.
        """
        return ExpPoly(self.omega_t, -const(self.omega) * (t1 - t2) ** 2)

    def __prior__(self):
        # Make parameters learnable:
        self.noise = self.ps.positive(self.noise, name="noise")
        self.alpha = self.ps.positive(self.alpha, name="alpha")
        self.alpha_t = self.ps.positive(self.alpha_t, name="alpha_t")
        self.gamma = self.ps.positive(self.gamma, name="gamma")
        self.omega = self.ps.positive(self.omega, name="omega")
        self.t_u = self.ps.unbounded(self.t_u, name="t_u")
        self.t_z = self.ps.unbounded(self.t_z, name="t_z")

        AbstractGPCM.__prior__(self)


class CGPCM(GPCM):
    """CGPCM variant of the GPCM.

    Takes in the same keyword arguments as :class:`.gpcm.GPCM`, but the keyword
    `causal` defaults to `True`.
    """

    name = "CGPCM"
    """str: Formatted name."""

    def __init__(self, causal=True, **kw_args):
        GPCM.__init__(self, causal=causal, **kw_args)


@method(GPCM)
def compute_K_u(model):
    """Covariance matrix of inducing variables :math:`u` associated with
    :math:`h`.

    Args:
        model (:class:`.gpcm.GPCM`): Model.

    Returns:
        tensor: :math:`K_u`.
    """
    return Dense(
        model.k_h(var("t1"), var("t2")).eval(
            t1=model.t_u[:, None], t2=model.t_u[None, :]
        )
    )


@method(GPCM)
def compute_K_z(model):
    """Covariance matrix :math:`K_z`.

    Args:
        model (:class:`.gpcm.GPCM`): Model.

    Returns:
        matrix: :math:`K_z`.
    """
    exppoly = model.k_xs(var("t1"), var("tau")) * model.k_xs(var("tau"), var("t2"))
    return Dense(exppoly.integrate("tau", t1=model.t_z[:, None], t2=model.t_z[None, :]))


@method(GPCM)
def compute_i_hx(model, t1=None, t2=None):
    """Compute the :math:`I_{hx}` integral.

    Args:
        model (:class:`.gpcm.GPCM`): Model.
        t1 (tensor, optional): First time input. Defaults to zero.
        t2 (tensor, optional): Second time input. Defaults to zero.

    Returns:
        tensor: Value of :math:`I_{hx}` for all `t1` and `t2`.
    """
    if t1 is None:
        t1 = B.zero(model.dtype)
    if t2 is None:
        t2 = B.zero(model.dtype)

    exppoly = model.k_h(var("t1") - var("tau"), var("t2") - var("tau"))

    if model.causal:
        upper = var("min_t1_t2")
    else:
        upper = np.inf

    return exppoly.integrate_box(
        ("tau", -np.inf, upper), t1=t1, t2=t2, min_t1_t2=B.minimum(t1, t2)
    )


@method(GPCM)
def compute_I_ux(model, t1=None, t2=None):
    """Compute the :math:`I_{ux}` integral.

    Args:
        model (:class:`.gpcm.GPCM`): Model.
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

    exppoly = model.k_h(var("t1") - var("tau"), var("t_u_1"))
    exppoly = exppoly * model.k_h(var("t_u_2"), var("t2") - var("tau"))

    if model.causal:
        upper = var("min_t1_t2")
    else:
        upper = np.inf

    result = exppoly.integrate_box(
        ("tau", -np.inf, upper),
        t1=t1,
        t2=t2,
        t_u_1=t_u_1,
        t_u_2=t_u_2,
        min_t1_t2=B.minimum(t1, t2),
    )

    if squeeze_t1 and squeeze_t2:
        return result[0, 0, :, :]
    elif squeeze_t1:
        return result[0, :, :, :]
    elif squeeze_t2:
        return result[:, 0, :, :]
    else:
        return result


@method(GPCM)
def compute_I_hz(model, t):
    """Compute the :math:`I_{hz,t_i}` matrix for :math:`t_i` in `t`.

    Args:
        model (:class:`.gpcm.GPCM`): Model.
        t (vector): Time points of data.

    Returns:
        tensor: Value of :math:`I_{hz,t_i}`, with shape
            `(len(t), len(model.t_z), len(model.t_z))`.
    """
    t = t[:, None, None]
    t_z_1 = model.t_z[None, :, None]
    t_z_2 = model.t_z[None, None, :]

    expq = (
        model.k_h(var("t") - var("tau1"), var("t") - var("tau2"))
        * model.k_xs(var("tau1"), var("t_z_1"))
        * model.k_xs(var("t_z_2"), var("tau2"))
    )

    if model.causal:
        upper1 = var("t")
        upper2 = var("t")
    else:
        upper1 = np.inf
        upper2 = np.inf

    return expq.integrate_box(
        ("tau1", -np.inf, upper1),
        ("tau2", -np.inf, upper2),
        t=t,
        t_z_1=t_z_1,
        t_z_2=t_z_2,
    )


@method(GPCM)
def compute_I_uz(model, t):
    """Compute the :math:`I_{uz,t_i}` matrix for :math:`t_i` in `t`.

    Args:
        model (:class:`.gpcm.GPCM`): Model.
        t (vector): Time points :math:`t_i` of data.

    Returns:
        tensor: Value of :math:`I_{uz,t_i}`, with shape
            `(len(t), len(model.t_u), len(model.t_z))`.
    """
    t = t[:, None, None]
    t_u = model.t_u[None, :, None]
    t_z = model.t_z[None, None, :]

    exppoly = model.k_h(var("t") - var("tau"), var("t_u"))
    exppoly = exppoly * model.k_xs(var("tau"), var("t_z"))

    if model.causal:
        upper = var("t")
    else:
        upper = np.inf

    return exppoly.integrate_box(("tau", -np.inf, upper), t=t, t_u=t_u, t_z=t_z)
