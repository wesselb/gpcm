import numpy as np
import pytest
from gpcm.exppoly import var, ExpPoly

from .util import approx


@pytest.fixture()
def t1():
    return var('t1')


@pytest.fixture()
def t2():
    return var('t2')


@pytest.fixture()
def t3():
    return var('t3')


@pytest.fixture()
def ep1(t1, t2, t3):
    return ExpPoly(4 - t1**2 - 2*t2**2 - 0.5*t1*t2 - 2*t1*t3 + 3*t2)


@pytest.fixture()
def ep2(t1):
    return ExpPoly(4 - t1**2 - 0.5*t1)


def test_case1(ep1):
    ref = np.array([[55.81808295, 11.76773162],
                    [11.76773162, 55.81808295]])
    res = ep1.integrate_box(('t1', -np.inf, 0),
                            ('t2', -np.inf, 0),
                            t3=np.eye(2))
    approx(res, ref, decimal=6)


def test_case2(ep1, t3):
    ref = np.array([[217.3921457, 318.3540954],
                    [318.3540954, 217.3921457]])
    res = ep1.integrate_box(('t1', -1, 2),
                            ('t2', t3, 3),
                            t3=np.eye(2))
    approx(res, ref, decimal=5)


def test3(ep2):
    ref = 65.73974603
    res = ep2.integrate_half('t1')
    approx(res, ref, decimal=6)
