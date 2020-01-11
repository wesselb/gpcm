import pytest
import numpy as np
from scipy.integrate import dblquad
from itertools import product
from gpcm.gprv import integral_abcd, integral_abcd_lu

from .util import approx


def signed_pairs(num):
    return [(np.abs(np.random.randn()), -np.abs(np.random.randn()))
            for _ in range(num)]


def test_integral_abcd():
    def integral_abcd_quadrature(a, b, c, d):
        return dblquad(lambda tau, tau2: np.exp(c*(tau + tau2) -
                                                d*np.abs(tau - tau2)),
                       0, a, lambda tau: 0, lambda tau: b)[0]

    for a, b, c, d in product(*signed_pairs(4)):
        approx(integral_abcd(a, b, c, d),
               integral_abcd_quadrature(a, b, c, d),
               decimal=5)


def test_integral_abcd_lu():
    def integral_abcd_lu_quadrature(a_lb, a_ub, b_lb, b_ub, c, d):
        return dblquad(lambda tau, tau2: np.exp(c*(tau + tau2) -
                                                d*np.abs(tau - tau2)),
                       a_lb, a_ub, lambda tau: b_lb, lambda tau: b_ub)[0]

    for a_lb, a_ub, b_lb, b_ub, c, d in product(*signed_pairs(6)):
        approx(integral_abcd_lu(a_lb, a_ub, b_lb, b_ub, c, d),
               integral_abcd_lu_quadrature(a_lb, a_ub, b_lb, b_ub, c, d),
               decimal=5)
