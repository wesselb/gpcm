from matrix import AbstractMatrix
from plum import convert
from stheno import Normal

__all__ = ["WhitenedNaturalNormal"]


class WhitenedNaturalNormal:
    def __init__(self, lam, prec, prior_cov):
        self.lam = lam
        self.prec = convert(prec, AbstractMatrix)
        self.prior_cov = prior_cov

        self._m2_hat = None

    def _inv_prec(self, a, both_sides=False):
        a = B.cholsolve(B.chol(self.prec), a)
        if both_sides:
            a = B.cholsolve(B.chol(self.prec), B.transpose(a))
        return a

    @property
    def m2_hat(self):
        if self._m2_hat is None:
            self.m2_hat = self._inv_prec(self.prec + B.outer(self.lam), both_sides=True)
        return self._m2_hat

    def sample_hat(self, state, num=1):
        state, noise = Normal(self.prec).sample(state, num)
        return B.cholsolve(B.chol(self.prec), noise + self.lam)
