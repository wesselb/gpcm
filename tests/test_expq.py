import numpy as np
import pytest

from gpcm.expq import var, const, EQ
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
def expq1(t1, t2, t3):
    return EQ(-const(1)*t1**2 +
              -const(2)*t2**2 +
              -const(.5)*t1*t2 +
              -const(2)*t1*t3 +
              const(3)*t2 +
              const(4))


@pytest.fixture()
def expq2(t1):
    return EQ(const(-1)*t1**2 +
              const(-.5)*t1 +
              const(4))


def test_case1(expq1):
    ref = np.array([[55.81808295, 11.76773162],
                    [11.76773162, 55.81808295]])
    res = expq1.integrate_box(('t1', -np.inf, 0),
                              ('t2', -np.inf, 0),
                              t3=np.eye(2))
    approx(res, ref, decimal=6)


def test_case2(expq1, t3):
    ref = np.array([[217.3921457, 318.3540954],
                    [318.3540954, 217.3921457]])
    res = expq1.integrate_box(('t1', const(-1), const(2)),
                              ('t2', t3, const(3)),
                              t3=np.eye(2))
    approx(res, ref, decimal=5)


def test3(expq2):
    ref = 65.73974603
    res = expq2.integrate_half('t1')
    approx(res, ref)
