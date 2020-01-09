import numpy as np
from lab import B
from scipy.integrate import quad

__all__ = ['ihx',
           'iux',
           'integral_abcd',
           'integral_abcd_lu',
           'I_hz',
           'I_ux',
           'I_uz']


def ihx(model, t1=0, t2=0):
    """Compute the :math:`I_{hx}` integral from the paper.

    Args:
        model (:class:`.model.GPRV`): Model.
        t1 (tensor, optional): First time input. Defaults to zero.
        t2 (tensor, optional): Second time input. Defaults to zero.

    Returns:
        tensor: Value of :math:`I_{hx}` for all `t1` and `t2`.
    """
    alpha_t, alpha, lam = model.alpha_t, model.alpha, model.lam
    return alpha_t**2/2/alpha*B.exp(-lam*B.abs(t1 - t2))


def I_hz(model, t):
    """Compute the :math:`I_{hz,t_i}` matrix for :math:`t_i` in t_vec.

    Args:
        model (:class:`.model.GPRV`): Model.
        t (vector): Time points of data.

    Returns:
        tensor: Value of :math:`I_{hz,t_i}`, with shape
            `(len(model.ms), len(model.ms), len(t))`.
    """
    # Compute sorting permutation.
    perm = np.argsort(model.ms)
    inverse_perm = np.arange(len(model.ms))
    for i, p in enumerate(perm):
        inverse_perm[p] = i

    # Sort to allow for simple concatenation.
    m_max = model.m_max
    ms = model.ms[perm]

    # Construct I_hz for m,n <= M.
    n_vec = ms[ms <= m_max]
    I_0_cos_1 = _I_hx_0_cos(model,
                            -n_vec[:, None, None] + n_vec[None, :, None],
                            t[None, None, :])
    I_0_cos_2 = _I_hx_0_cos(model,
                            n_vec[:, None, None] + n_vec[None, :, None],
                            t[None, None, :])
    I_hz_mnleM = 0.5*(I_0_cos_1 + I_0_cos_2)

    # Construct I_hz for m,n > M.
    n_vec = ms[ms > m_max] - m_max
    I_0_cos_1 = _I_hx_0_cos(model,
                            -n_vec[:, None, None] + n_vec[None, :, None],
                            t[None, None, :])
    I_0_cos_2 = _I_hx_0_cos(model,
                            n_vec[:, None, None] + n_vec[None, :, None],
                            t[None, None, :])
    I_hz_mngtM = 0.5*(I_0_cos_1 - I_0_cos_2)

    # Construct I_hz for 0 < m <= M and n > M.
    n_vec = ms[(0 < ms)*(ms <= m_max)]
    n_vec_2 = ms[ms > m_max]  # Do not subtract M!
    I_0_sin_1 = _I_hx_0_sin(model,
                            n_vec[:, None, None] + n_vec_2[None, :, None],
                            t[None, None, :])
    I_0_sin_2 = _I_hx_0_sin(model,
                            -n_vec[:, None, None] + n_vec_2[None, :, None],
                            t[None, None, :])
    I_hz_mleM_ngtM = 0.5*(I_0_sin_1 + I_0_sin_2)

    # Construct I_hz for m = 0 and n > M.
    n_vec = ms[ms == 0]
    n_vec_2 = ms[ms > m_max]  # Do not subtract M!
    I_hz_0_gtM = _I_hx_0_sin(model,
                             n_vec[:, None, None] + n_vec_2[None, :, None],
                             t[None, None, :])

    # Concatenate to form I_hz for m <= M and n > M.
    I_hz_mleM_ngtM = B.concat(I_hz_0_gtM, I_hz_mleM_ngtM, axis=0)

    # Compute the other half by transposing.
    I_hz_mgtM_nleM = B.transpose(I_hz_mleM_ngtM, perm=(1, 0, 2))

    # Construct result.
    result = B.concat2d([I_hz_mnleM, I_hz_mleM_ngtM],
                        [I_hz_mgtM_nleM, I_hz_mngtM])

    # Undo sorting and return.
    result = B.take(result, inverse_perm, axis=0)
    result = B.take(result, inverse_perm, axis=1)
    return result


def _I_hx_0_cos(model, n, t):
    """Compute the for :math:`I_{0,n:\cos}` integral."""
    omega_n = 2*B.pi*n/(model.b - model.a)
    t_less_a = t - model.a
    return (model.alpha_t**2/(4*model.alpha**2 + omega_n**2)*
            (2*model.alpha*(B.cos(omega_n*t_less_a) -
                            B.exp(-2*model.alpha*t_less_a)) +
             omega_n*B.sin(omega_n*t_less_a))) + \
           (model.alpha_t**2/(2*(model.alpha + model.lam))*
            B.exp(-2*model.alpha*t_less_a))


def _I_hx_0_sin(model, n, t):
    """Compute the :math:`I_{0,n:\sin}`, :math:`n>M`, integral."""
    omega_vec = 2*B.pi*(n - model.m_max)/(model.b - model.a)
    t_less_a = t - model.a
    return (model.alpha_t**2/(4*model.alpha**2 + omega_vec**2)*
            (omega_vec*(B.exp(-2*model.alpha*t_less_a) -
                        B.cos(omega_vec*t_less_a)) +
             2*model.alpha*B.sin(omega_vec*t_less_a)))


def I_ux(model):
    """Compute the :math:`I_{ux}` matrix.

    Args:
        model (:class:`.model.GPRV`): Model.

    Returns:
        tensor: Value of :math:`I_{ux}`, with shape
            `(len(model.t_u), len(model.t_u))`.
    """
    t_u = B.flatten(model.t_u)
    return iux(model,
               t1=B.cast(model.dtype, 0),
               t2=B.cast(model.dtype, 0),
               t_u_1=t_u[:, None],
               t_u_2=t_u[None, :])


def iux(model, t1, t2, t_u_1, t_u_2):
    """Compute the :math:`I_{ux}` integral from the paper.

    Args:
        model (:class:`.model.GPRV`): Model.
        t1 (tensor): First time input.
        t2 (tensor): Second time input.
        t_u_1 (tensor): First inducing point location input.
        t_u_2 (tensor): Second inducing point location input.

    Returns:
        tensor: Value of :math:`I_{hx}` for all `t1`, `t2`, `tu1`, and `tu2`.
    """
    ag = model.gamma - model.alpha
    return model.alpha_t**2*model.gamma_t**2* \
           B.exp(-model.gamma*(t_u_1 + t_u_2) + ag*(t1 + t2))* \
           integral_abcd_lu(-t1, t_u_1 - t1, -t2, t_u_2 - t2, ag, model.lam)


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
    condition = a*b >= 0

    # Compute the two parts.
    part1 = sign*d/c*(1 - B.exp(2*c*sign*B.minimum(B.abs(a), B.abs(b))))
    part2 = 1 - \
            B.exp(c*a - d*B.abs(a)) - \
            B.exp(c*b - d*B.abs(b)) + \
            B.exp(c*(a + b) - d*B.abs(a - b))

    # Combine and return.
    condition = B.cast(B.dtype(part1), condition)
    return (condition*part1 + part2)/(c**2 - d**2)


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
    return integral_abcd(a_ub, b_ub, c, d) + \
           integral_abcd(a_lb, b_lb, c, d) - \
           integral_abcd(a_ub, b_lb, c, d) - \
           integral_abcd(a_lb, b_ub, c, d)


def I_uz(model, t):
    """Compute the :math:`I_{uz,t_i}` matrix for :math:`t_i` in `t`.

    Args:
        model (:class:`.model.GPRV`): Model.
        t (vector): Time points :math:`t_i` of data.

    Returns:
        tensor: Value of :math:`I_{uz,t_i}`, with shape
            `(len(model.t_u), len(model.ms), len(t))`.
    """
    # Compute sorting permutation.
    perm = np.argsort(model.ms)
    inverse_perm = np.arange(len(model.ms))
    for i, p in enumerate(perm):
        inverse_perm[p] = i

    # Sort to allow for simple concatenation.
    ms = model.ms[perm]

    factor = (model.alpha_t*model.gamma_t*
              B.exp(-model.gamma*model.t_u[:, None, None]))

    # Compute I(l, u, 0).
    k = ms[ms == 0]
    I_uz0 = factor*_integral_lu0(model,
                                 t[None, None, :] -
                                 model.t_u[:, None, None],
                                 t[None, None, :]) + \
            k[None, :, None]

    # Compute I(l, u, k) for 0 < k < M + 1.
    k = ms[(0 < ms)*(ms <= model.m_max)]
    I_uz_leM = factor*_integral_luk_leq_M(model,
                                          t[None, None, :] -
                                          model.t_u[:, None, None],
                                          t[None, None, :],
                                          k[None, :, None])

    # Compute I(l, u, k) for k > M.
    k = ms[ms > model.m_max]
    I_uz_gtM = factor*_integral_luk_g_M(model,
                                        t[None, None, :] -
                                        model.t_u[:, None, None],
                                        t[None, None, :],
                                        k[None, :, None])

    # Construct result.
    result = B.concat(I_uz0, I_uz_leM, I_uz_gtM, axis=1)

    # Undo sorting and return.
    return B.take(result, inverse_perm, axis=1)


def _integral_lu0(model, l, u):
    """Compute :math:`I(l,u,k)` for :math:`k=0`."""
    ag = model.alpha - model.gamma
    return 1/ag*(1 - B.exp(ag*(l - u)))


def _integral_luk_leq_M(model, l, u, k):
    """Compute :math:`I(l,u,k)` for :math:`0<k<M+1`. Assumes that
    :math:`a<l,u<b`.
    """
    ag = model.alpha - model.gamma
    om = 2*B.pi*k/(model.b - model.a)
    return 1/(ag**2 + om**2)* \
           ((ag*B.cos(om*(u - model.a)) + om*B.sin(om*(u - model.a))) -
            B.exp(ag*(l - u))*
            (ag*B.cos(om*(l - model.a)) + om*B.sin(om*(l - model.a))))


def _integral_luk_g_M(model, l, u, k):
    """Internal function :math:`I(l,u,k)` for :math:`k>M`. Assumes
    that :math:`a<l,u<b`.
    
    Note:
        This function gives :math:`-I(l,u,k)`, i.e. *minus* the integral in the
        paper!
    """
    ag = model.alpha - model.gamma
    om = 2*B.pi*(k - model.m_max)/(model.b - model.a)
    return -1/(ag**2 + om**2)* \
           ((-ag*B.sin(om*(u - model.a)) + om*B.cos(om*(u - model.a))) -
            B.exp(ag*(l - u))*
            (-ag*B.sin(om*(l - model.a)) + om*B.cos(om*(l - model.a))))


def I_hz_quad(model, t_1, t_2=None):
    """Compute the :math:`I_{hz}(t, t')` matrix with **quadrature** for
    :math:`t` in `t_1` and :math:`t'` in `t_2`.

    Args:
        model (:class:`.model.GPRV`): Model.
        t_1 (vector): Time points :math:`t` of data.
        t_2 (vector, optional): Other time points :math:`t'` of data.
            Defaults to `t_vec_1`.

    Returns:
        tensor: Value of :math:`I_{hz}(t, t')`, with shape
            `(len(model.ms), len(model.ms), len(t_vec_1), len(t_vec_2)`.
    """
    # Set default for `t_vec_2`.
    if t_2 is None:
        symmetric = True
        t_2 = t_1
    else:
        symmetric = False

    # Construct output.
    out = B.zeros(model.ms.size, model.ms.size, t_1.size, t_2.size)

    # Fill output, exploiting symmetry.
    if not symmetric:
        it = np.nditer(out, flags=['multi_index'])
        while not it.finished:
            i, j, k, l = it.multi_index
            out[i, j, k, l] = ihz_quad(model,
                                       m=model.ms[i],
                                       n=model.ms[j],
                                       t_1=t_1[k],
                                       t_2=t_2[l])
            it.iternext()
    else:
        for i in range(model.ms.size):
            for j in range(i, model.ms.size):
                for k in range(t_1.size):
                    for l in range(t_2.size):
                        out[i, j, k, l] = ihz_quad(model,
                                                   m=model.ms[i],
                                                   n=model.ms[j],
                                                   t_1=t_1[k],
                                                   t_2=t_2[l])
                if i is not j:
                    out[j, i, :, :] = out[i, j, :, :].T
    return out


def ihz_quad(model, m, n, t_1, t_2):
    """Compute the :math:`I_{hz}(t,t')` integral for `t_1` and `t_2`.
    :math:`t'` in `t_2`.

    Args:
        model (:class:`.model.GPRV`): Model.
        m (int): Frequency corresponding to `t_1`.
        n (int): Frequency corresponding to `t_2`.
        t_1 (scalar): :math:`t` of data point.
        t_2 (scalar, optional): :math:`t'` of data point. Defaults to `t_1`.

    Returns:
        scalar: Value of :math:`I_{hz}(t, t')`.
    """

    # Ensure that `t_1 <= t_2`.
    if t_1 > t_2:
        t_1, t_2 = t_2, t_1
        m, n = n, m

    if m <= model.m_max and n <= model.m_max:
        int_left_a = (model.alpha_t**2/(2*(model.alpha + model.lam))*
                      np.exp(-2*model.alpha*(t_2 - model.a))*
                      np.exp(-model.lam*(t_2 - t_1)))
    else:
        int_left_a = 0

    def window(tau):
        return model.alpha_t*np.exp(-model.alpha*np.abs(tau))

    def beta(tau, m):
        if m <= model.m_max:
            left_a = np.exp(-model.lam*(tau - model.b))
            right_b = np.exp(model.lam*(tau - model.a))
            omega = 2*np.pi*m/(model.b - model.a)
            in_a_b = np.cos(omega*(tau - model.a))
        else:
            left_a = 0
            right_b = 0
            omega = 2*np.pi*(m - model.m_max)/(model.b - model.a)
            in_a_b = np.sin(omega*(tau - model.a))
        return (left_a*(tau > model.b) +
                right_b*(tau < model.a) +
                in_a_b*(tau >= model.a)*(tau <= model.b))

    def f(tau):
        return window(t_2 - tau)**2*beta(tau - t_2 + t_1, m)*beta(tau, n)

    return quad(f, model.a, t_2)[0] + int_left_a
