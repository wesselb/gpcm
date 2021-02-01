import pytest
import numpy as np
from scipy.integrate import dblquad, quad
from itertools import product
import gpcm.gprv as gprv
from varz import Vars
import lab as B

from .util import approx, assert_positive_definite


@pytest.fixture()
def t():
    return B.linspace(0, 2, 5)


@pytest.fixture()
def model(t):
    return gprv.GPRV(Vars(np.float64), window=0.5, t=t, n_u=3, m_max=2)


def signed_pairs(num):
    return [(np.abs(np.random.randn()), -np.abs(np.random.randn())) for _ in range(num)]


def test_K_u(model):
    assert_positive_definite(model.compute_K_u())
    approx(gprv.compute_K_u(model)[0, 0], 1)  # Test default of `gamma_t`.


def test_K_z(model):
    assert_positive_definite(model.compute_K_z())


def test_i_hx(model, t):
    assert_positive_definite(model.compute_i_hx(t[:, None], t[None, :]))
    approx(model.compute_i_hx(1, 1), 1)  # Test default for `alpha_t`.


def test_integral_abcd():
    def integral_quadrature(a, b, c, d):
        return dblquad(
            lambda tau, tau2: np.exp(c * (tau + tau2) - d * np.abs(tau - tau2)),
            0,
            a,
            lambda tau: 0,
            lambda tau: b,
        )[0]

    for a, b, c, d in product(*signed_pairs(4)):
        approx(
            gprv.integral_abcd(a, b, c, d), integral_quadrature(a, b, c, d), atol=1e-5
        )


def test_integral_abcd_lu():
    def integral_quadrature(a_lb, a_ub, b_lb, b_ub, c, d):
        return dblquad(
            lambda tau, tau2: np.exp(c * (tau + tau2) - d * np.abs(tau - tau2)),
            a_lb,
            a_ub,
            lambda tau: b_lb,
            lambda tau: b_ub,
        )[0]

    for a_lb, a_ub, b_lb, b_ub, c, d in product(*signed_pairs(6)):
        approx(
            gprv.integral_abcd_lu(a_lb, a_ub, b_lb, b_ub, c, d),
            integral_quadrature(a_lb, a_ub, b_lb, b_ub, c, d),
            atol=1e-5,
        )


def test_I_ux(model, t):
    def integral_quadrature(t1, t2, t_u_1, t_u_2):
        def integral(tau1, tau2):
            return (
                model.alpha_t ** 2
                * model.gamma_t ** 2
                * B.exp(
                    -model.alpha * (tau1 + tau2)
                    + -model.gamma * (t_u_1 - tau1)
                    + -model.gamma * (t_u_2 - tau2)
                    + -model.lam * B.abs((t1 - tau1) - (t2 - tau2))
                )
            )

        return dblquad(integral, 0, t_u_1, lambda tau: 0, lambda tau: t_u_2)[0]

    I_ux = model.compute_I_ux(t, t)

    for i in range(len(t)):
        for j in range(len(t)):
            for k in range(len(model.t_u)):
                for l in range(len(model.t_u)):
                    approx(
                        integral_quadrature(t[i], t[j], model.t_u[k], model.t_u[l]),
                        I_ux[i, j, k, l],
                        atol=1e-5,
                    )


def beta(model, m, tau):
    a = model.a
    b = model.b
    m_max = model.m_max

    if a < tau < b:
        if m <= m_max:
            return B.cos(2 * B.pi * m / (b - a) * (tau - a))
        else:
            return B.sin(2 * B.pi * (m - m_max) / (b - a) * (tau - a))
    elif tau <= a:
        if m <= m_max:
            return B.exp(-model.lam * (a - tau))
        else:
            return 0
    else:
        if m <= m_max:
            return B.exp(-model.lam * (tau - b))
        else:
            return 0


def test_I_hz(model, t):
    def integral_quadrature(t, m, n):
        def integral(tau):
            return (
                model.alpha_t ** 2
                * B.exp(-2 * model.alpha * B.abs(t - tau))
                * beta(model, m, tau)
                * beta(model, n, tau)
            )

        return quad(integral, -np.inf, t)[0]

    I_hz = model.compute_I_hz(t)

    for i in range(len(t)):
        for j in range(len(model.ms)):
            for k in range(len(model.ms)):
                approx(
                    I_hz[i, j, k],
                    integral_quadrature(t[i], model.ms[j], model.ms[k]),
                    atol=1e-5,
                )


def test_I_uz(model, t):
    def integral_quadrature(t, t_u, m):
        def integral(tau):
            return (
                model.alpha_t
                * model.gamma_t
                * B.exp(-model.alpha * B.abs(tau) + -model.gamma * B.abs(t_u - tau))
                * beta(model, m, t - tau)
            )

        return quad(integral, 0, t_u)[0]

    I_uz = model.compute_I_uz(t)

    for i in range(len(t)):
        for j in range(len(model.t_u)):
            for k in range(len(model.ms)):
                approx(
                    I_uz[i, j, k],
                    integral_quadrature(t[i], model.t_u[j], model.ms[k]),
                    atol=1e-5,
                )
