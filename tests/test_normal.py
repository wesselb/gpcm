import lab as B
from gpcm.normal import NaturalNormal
from stheno import Normal

from .util import approx


def test_natural_normal():
    chol = B.randn(3, 3)
    dist = Normal(B.randn(3, 1), B.reg(chol @ chol.T, diag=1e-1))
    nat = NaturalNormal.from_normal(dist)

    # Test properties.
    assert dist.dtype == nat.dtype
    for name in ["dim", "mean", "var", "m2"]:
        approx(getattr(dist, name), getattr(nat, name))

    # Test sampling.
    state = B.create_random_state(dist.dtype, seed=0)
    state, sample = nat.sample(state, num=1_000_000)
    emp_mean = B.mean(B.dense(sample), axis=1, squeeze=False)
    emp_var = (sample - emp_mean) @ (sample - emp_mean).T / 1_000_000
    approx(dist.mean, emp_mean, rtol=5e-2)
    approx(dist.var, emp_var, rtol=5e-2)

    # Test KL.
    chol = B.randn(3, 3)
    other_dist = Normal(B.randn(3, 1), B.reg(chol @ chol.T, diag=1e-2))
    other_nat = NaturalNormal.from_normal(other_dist)
    approx(dist.kl(other_dist), nat.kl(other_nat))

    # Test log-pdf.
    x = B.randn(3, 1)
    approx(dist.logpdf(x), nat.logpdf(x))
