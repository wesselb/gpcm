from gpcm.expq import EQ, const

__all__ = []


def k_h_constructor(alpha, gamma):
    """Create a function that calls `k_h` for some given hyperparameters.

    Args:
        alpha (scalar): Kernel decay parameter.
        gamma (scalar): Kernel length scale.

    Returns:
        function: Constructor.
    """

    def constructor(x, y):
        return k_h(alpha, gamma, x, y)

    return constructor


def k_h(alpha, gamma, t1, t2):
    """Kernel of :math:`h` process.

    Args:
        alpha (scalar): Kernel decay parameter.
        gamma (scalar): Kernel length scale.
        t1 (:class:`.expq.Poly`): First input.
        t2 (:class:`.expq.Poly`): Second input.

    Returns:
        :class:`.expq.EQ`: Resulting exponentiated quadratic form.
    """
    return EQ(-const(alpha)*(t1**2 + t2**2) - const(gamma)*(t1 - t2)**2)


def k_xs_constructor(omega):
    """Create a function that calls `k_xs` for some given hyperparameter.

    Args:
        omega (scalar): Kernel length scale parameter.

    Returns:
        function: Constructor.
    """

    def constructor(x, y):
        return k_xs(omega, x, y)

    return constructor


def k_xs(omega, t1, t2):
    """Kernel between :math:`s` and :math:`x` processes.

    Args:
        omega (scalar): Kernel length scale parameter.
        t1 (:class:`.expq.Poly`): First input.
        t2 (:class:`.expq.Poly`): Second input.

    Returns:
        :class:`.expq.EQ`: Resulting exponentiated quadratic form.
    """
    return EQ(-const(omega)*(t1 - t2)**2)
