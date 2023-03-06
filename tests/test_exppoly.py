import numpy as np
import pytest
from gpcm.exppoly import (
    ExpPoly,
    Factor,
    Poly,
    Term,
    _as_poly,
    _merge_common_factors,
    _merge_common_terms,
    const,
    is_inf,
    safe_sqrt,
    var,
)

from .util import approx


@pytest.fixture()
def t1():
    return var("t1")


@pytest.fixture()
def t2():
    return var("t2")


@pytest.fixture()
def t3():
    return var("t3")


def test_safe_sqrt():
    assert safe_sqrt(4.0) == 2
    assert safe_sqrt(0.0) == 1e-15
    assert safe_sqrt(0) == 0


def test_is_inf():
    assert not is_inf(None)
    assert not is_inf(1)
    assert is_inf(np.inf)
    assert is_inf(-np.inf)


def test_const():
    assert isinstance(const(1), Poly)


def test_var():
    assert isinstance(var("t"), Poly)


def test_factor_constructor():
    assert Factor("t") == Factor("t", 1)

    with pytest.raises(ValueError):
        Factor("t", 0)


def test_factor_str():
    f = Factor("t", 2)
    assert str(f) == "t^2"
    assert str(f) == repr(f)


def test_factor_eval():
    assert Factor("t", 2).eval(t=2) == 4


@pytest.mark.parametrize("f", [lambda x: x, hash])
def test_factor_equality(f):
    assert f(Factor("t", 2)) == f(Factor("t", 2))
    assert f(Factor("t", 2)) != f(Factor("t2", 2))
    assert f(Factor("t", 1)) != f(Factor("t", 2))


def test_factor_multiply():
    assert Factor("t", 1) * Factor("t", 2) == Factor("t", 3)

    f = Factor("t", 1)
    assert 1 * f is f
    assert f * 1 is f

    with pytest.raises(RuntimeError):
        Factor("t1", 1) * Factor("t2", 1)
    with pytest.raises(RuntimeError):
        Factor("t", 1) * 2


def test_merge_common_factors():
    result = set(
        _merge_common_factors(
            Factor("t1", 1),
            Factor("t2", 2),
            Factor("t3", 3),
            Factor("t2", 4),
            Factor("t1", 5),
        )
    )
    expected = {Factor("t1", 6), Factor("t2", 6), Factor("t3", 3)}
    assert result == expected


def test_term_constructor():
    assert Term(Factor("t", 2)) == Term(1, Factor("t", 2))
    assert Term(Factor("t", 2), Factor("t", 3)) == Term(Factor("t", 5))
    assert Term(0, Factor("t", 2)) == Term(0)


def test_term_is_function_of():
    t = Term(2, Factor("t1", 1), Factor("t2", 1))
    assert t.is_function_of("t1")
    assert t.is_function_of("t2")
    assert not t.is_function_of("t3")


def test_term_collect_for():
    t = Term(2, Factor("t1", 1), Factor("t2", 2))
    assert t.collect_for(Factor("t1", 1)) == Term(2, Factor("t2", 2))
    assert t.collect_for(Factor("t2", 2)) == Term(2, Factor("t1", 1))
    with pytest.raises(RuntimeError):
        t.collect_for(Factor("t3", 2))


def test_term_eval():
    assert Term(2, Factor("t", 3)).eval(t=4) == 2 * 4**3


def test_term_is_constant():
    assert Term(2).is_constant()
    assert not Term(2, Factor("t", 1)).is_constant()


def test_term_highest_power():
    t = Term(2, Factor("t1", 2), Factor("t2", 3))
    assert t.highest_power("t1") == 2
    assert t.highest_power("t2") == 3
    assert t.highest_power("t3") == 0


def test_term_substitute(t1, t2):
    result = Term(2, Factor("t1", 1), Factor("t2", 2)).substitute("t2", t1 - t2)
    expected = 2 * t1**3 - 4 * t1**2 * t2 + 2 * t1 * t2**2
    assert result == expected


def test_term_str():
    t = Term(2, Factor("t2", 2), Factor("t1", 1))
    assert str(t) == "2 * t1^1 t2^2"
    assert str(t) == repr(t)

    assert str(Term(1)) == "1"


@pytest.mark.parametrize("f", [lambda x: x, hash])
def test_term_equality(f):
    x = Term(2, Factor("t1", 1), Factor("t2", 2))
    y = Term(2, Factor("t2", 2), Factor("t1", 1))
    assert f(x) == f(y)

    x = Term(2, Factor("t1", 1), Factor("t2", 2))
    y = Term(2, Factor("t1", 2), Factor("t2", 2))
    assert f(x) != f(y)

    x = Term(1, Factor("t1", 1), Factor("t2", 2))
    y = Term(2, Factor("t1", 1), Factor("t2", 2))
    assert f(x) != f(y)


def test_term_addition():
    assert Term(1, Factor("t", 2)) + Term(2, Factor("t", 2)) == Term(3, Factor("t", 2))

    t = Term(1, Factor("t", 1))
    assert t + 0 is t
    assert 0 + t is t

    with pytest.raises(RuntimeError):
        Term(1, Factor("t1", 1)) + Term(1, Factor("t2", 1))
    with pytest.raises(RuntimeError):
        t + 1


def test_term_multiplication():
    result = Term(1, Factor("t1", 1)) * Term(2, Factor("t2", 2))
    expected = Term(2, Factor("t1", 1), Factor("t2", 2))
    assert result == expected

    t = Term(1, Factor("t", 1))
    assert t * 1 is t
    assert 1 * t is t

    with pytest.raises(RuntimeError):
        t * 2


def test_merge_common_terms():
    result = set(
        _merge_common_terms(
            Term(1, Factor("t1", 1)),
            Term(2, Factor("t2", 1)),
            Term(3, Factor("t3", 1)),
            Term(4, Factor("t2", 1)),
            Term(5, Factor("t1", 1)),
        )
    )
    assert result == {
        Term(6, Factor("t1", 1)),
        Term(6, Factor("t2", 1)),
        Term(3, Factor("t3", 1)),
    }


def test_poly_is_function_of(t1, t2):
    p = t1**2 * t2 + 1
    assert p.is_function_of("t1")
    assert p.is_function_of("t2")
    assert not p.is_function_of("t3")


def test_poly_collect_for(t1, t2, t3):
    result = (t1**2 * t2 * t3**3 + 2 * t1**2).collect_for(Factor("t1", 2))
    expected = 2 + t2 * t3**3
    assert result == expected


def test_poly_reject(t1, t2, t3):
    assert (t1**2 * t2 * t3**3 + 2 * t1 + 3 * t2**2).reject("t1") == 3 * t2**2


def test_poly_eval(t1, t2):
    assert (2 * t1**2 + 3 * t2).eval(t1=4, t2=5) == 2 * 4**2 + 3 * 5


def test_poly_is_constant(t1, t2):
    assert const(1).is_constant()
    assert not (t1 + const(1)).is_constant()


def test_poly_highest_power(t1, t2):
    p = t1**2 * t2 + t2**3 + 2
    assert p.highest_power("t1") == 2
    assert p.highest_power("t2") == 3
    assert p.highest_power("t3") == 0


def test_poly_substitute(t1, t2, t3):
    assert (2 * t3**2).substitute(
        "t3", t1 - t2
    ) == 2 * t1**2 - 4 * t1 * t2 + 2 * t2**2


def test_poly_str(t1, t2):
    p = 2 + t1 * t2
    assert str(p) in {"2 + 1 * t1^1 t2^1", "1 * t1^1 t2^1 + 2"}
    assert str(p) == repr(p)

    assert str(Poly()) == "0"


def test_poly_constructor():
    assert Poly(
        Term(1, Factor("t1", 1)), Term(2, Factor("t1", 1)), Term(2, Factor("t2", 2))
    ) == Poly(Term(3, Factor("t1", 1)), Term(2, Factor("t2", 2)))


def test_poly_equality(t1, t2, t3):
    assert t1 + t2 == t2 + t1
    assert t1 + t2 != t1 + t3
    assert 2 * (t1 + t2) == 2 * t1 + 2 * t2
    assert (t1 + t2) ** 2 == t1**2 + 2 * t1 * t2 + t2**2


def test_poly_addition(t1, t2, t3):
    assert Poly(Term(1, Factor("t1", 1))) + Poly(Term(1, Factor("t2", 1))) == Poly(
        Term(1, Factor("t1", 1)), Term(1, Factor("t2", 1))
    )
    assert t1 + 0 is t1
    assert 0 + t1 is t1
    assert t1 + 1 == t1 + const(1)
    assert 1 + t1 == t1 + const(1)
    assert -t1 == Poly(Term(-1, Factor("t1", 1)))


def test_poly_subtraction(t1, t2):
    assert t1 - t1 == Poly(Term(0))
    assert 1 - t1 == Poly(Term(1), Term(-1, Factor("t1", 1)))
    assert t1 - 1 == Poly(Term(-1), Term(1, Factor("t1", 1)))


def test_poly_multiplication(t1, t2, t3):
    assert Poly(Term(1, Factor("t1", 1))) * Poly(Term(2, Factor("t2", 1))) == Poly(
        Term(2, Factor("t1", 1), Factor("t2", 1))
    )
    assert t1 * 1 is t1
    assert 1 * t1 is t1
    assert t1 * 2 == t1 * const(2)
    assert 2 * t1 == t1 * const(2)


def test_poly_pow(t1):
    assert t1**3 == t1 * t1 * t1

    with pytest.raises(RuntimeError):
        t1**-2


def test_as_poly(t1):
    assert _as_poly(2) == const(2)
    assert _as_poly(t1) is t1


def test_exppoly_constructor(t1):
    assert ExpPoly(t1) == ExpPoly(1, t1)


def test_exppoly_substitute(t1, t2, t3):
    assert ExpPoly(2, -(t3**2)).substitute("t3", t1 - t2) == ExpPoly(
        2, -(t1**2) + 2 * t1 * t2 - t2**2
    )


def test_exppoly_eval(t1, t2):
    assert ExpPoly(2, -((t1 - t2) ** 3)).eval(t1=2, t2=4) == 2 * np.exp(-((-2) ** 3))


def test_exppoly_str(t1, t2):
    ep = ExpPoly(2, t1 + t2)
    assert str(ep) in {"2 * exp(1 * t1^1 + 1 * t2^1)", "2 * exp(1 * t2^1 + 1 * t1^1)"}
    assert str(ep) == repr(ep)


def test_exppoly_equality(t1, t2, t3):
    assert ExpPoly(2, t1) == ExpPoly(2, t1)
    assert ExpPoly(2, t1) != ExpPoly(2, t2)
    assert ExpPoly(1, t1) != ExpPoly(2, t1)


def test_exppoly_multiplication(t1, t2):
    assert ExpPoly(2, t1) * ExpPoly(3, t2) == ExpPoly(6, t1 + t2)
    assert ExpPoly(2, t1) * 2 == ExpPoly(4, t1)
    assert 2 * ExpPoly(2, t1) == ExpPoly(4, t1)
    assert -ExpPoly(2, t1) == ExpPoly(-2, t1)


def test_exppoly_integrate_quadratic_coefficient_check(t1, t2):
    with pytest.raises(RuntimeError):
        ExpPoly(t2 * t1**2).integrate("t1")


def test_exppoly_integrate_quadratic_dependency_check(t1, t2):
    with pytest.raises(RuntimeError):
        ExpPoly(t1)._integrate("t1")
    with pytest.raises(RuntimeError):
        ExpPoly(t1**3)._integrate("t1")

    with pytest.raises(RuntimeError):
        ExpPoly(t1)._integrate_half1("t1")
    with pytest.raises(RuntimeError):
        ExpPoly(t1**3)._integrate_half1("t1")

    with pytest.raises(RuntimeError):
        ExpPoly(t1 + t2**2)._integrate_half2("t1", "t2")
    with pytest.raises(RuntimeError):
        ExpPoly(t1**3 + t2**2)._integrate_half2("t1", "t2")
    with pytest.raises(RuntimeError):
        ExpPoly(t1**2 + t2)._integrate_half2("t1", "t2")
    with pytest.raises(RuntimeError):
        ExpPoly(t1**2 + t2**3)._integrate_half2("t1", "t2")


@pytest.fixture()
def ep1(t1):
    return ExpPoly(4 - t1**2 - 0.5 * t1)


@pytest.fixture()
def ep2(t1, t2, t3):
    return ExpPoly(4 - t1**2 - 2 * t2**2 - 0.5 * t1 * t2 - 2 * t1 * t3 + 3 * t2)


def test_exppoly_case1(ep1):
    ref = 103.0140042
    res = ep1.integrate("t1")
    approx(res, ref, atol=1e-6)


def test_exppoly_case2(ep1):
    ref = 65.73974603
    res = ep1.integrate_half("t1")
    approx(res, ref, atol=1e-6)


def test_exppoly_case3(ep2):
    ref = np.array([[1627.297351, 393.5943995], [393.5943995, 1627.297351]])
    res = ep2.integrate_box(
        ("t1", -np.inf, np.inf), ("t2", -np.inf, np.inf), t3=np.eye(2)
    )
    approx(res, ref, atol=1e-4)


def test_exppoly_case4(ep2):
    ref = np.array([[55.81808295, 11.76773162], [11.76773162, 55.81808295]])
    res = ep2.integrate_box(("t1", -np.inf, 0), ("t2", -np.inf, 0), t3=np.eye(2))
    approx(res, ref, atol=1e-6)


def test_exppoly_case5(ep2, t3):
    ref = np.array([[217.3921457, 318.3540954], [318.3540954, 217.3921457]])
    res = ep2.integrate_box(("t1", -1, 2), ("t2", t3, 3), t3=np.eye(2))
    approx(res, ref, atol=1e-5)
