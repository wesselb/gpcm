import lab as B


class GPRV:
    """GPRV variation of the GPCM.

    Args:
        lam (scalar): Decay of the kernel of :math:`x`.
        alpha (scalar): Decay of the window.
        alpha_t (scalar, optional): Scale of the window. Defaults to
            normalise the window to unity power.
        gamma (scalar, optional): Decay of the transform :math:`u(t)` of
            :math:`x`. Defaults to the inverse of twice the spacing between the
            inducing points.
        gamma_t (scalar, optional): Scale of the transform. Defaults to
            normalise the transform to unity power.
        a (scalar): Lower bound of support of the basis.
        b (scalar): Upper bound of support of the basis.
        m_max (scalar): Defines cosine and sine basis functions.
        ms (vector, optional): Basis function frequencies. Defaults to
            :math:`0,\ldots,2M-1`.
        n_u (int, optional): Number of inducing points of :math:`u(t_{u_i})`.
        t_u (vector, optional): Location :math:`t_{u_i}` of :math:`u(t_{u_i})`.
            Defaults to equally spaced points twice the filter length scale.
    """

    def __init__(self,
                 lam=None,
                 alpha=None,
                 alpha_t=None,
                 gamma=None,
                 gamma_t=None,
                 a=None,
                 b=None,
                 m_max=None,
                 ms=None,
                 n_u=None,
                 t_u=None):
        self.lam = lam
        self.alpha = alpha
        self.alpha_t = alpha_t
        self.a = a
        self.b = b
        self.m_max = m_max
        self.ms = ms
        self.n_u = n_u
        self.t_u = t_u
        self.gamma = gamma
        self.gamma_t = gamma_t

        self.dtype = B.dtype(self.lam)

        if self.alpha_t is None:
            self.alpha_t = B.sqrt(2*self.alpha)

        if self.ms is None:
            self.ms = B.range(2*self.m_max + 1)

        if self.t_u is None:
            self.t_u = B.linspace(0, 2/self.alpha, n_u)
            self.n_u = B.length(self.t_u)

        if self.gamma is None:
            self.gamma = 1/(2*(self.t_u[1] - self.t_u[0]))

        if self.gamma_t is None:
            self.gamma_t = B.sqrt(2*self.gamma)


def determine_a_b(alpha, t):
    """Determine parameters :math:`a` and :math:`b` for the GPRV model.

    Args:
        alpha (scalar): Decay of the window.
        t (vector): Time points of data.

    Returns:
        tuple: Tuple containing :math:`a` and :math:`b`.
    """
    return min(t) - 2/alpha, max(t)
