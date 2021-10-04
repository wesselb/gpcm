import lab as B
from matrix import AbstractMatrix, structured
from plum import Dispatcher, convert
from stheno import Normal

__all__ = ["NaturalNormal"]

_dispatch = Dispatcher()


class NaturalNormal:
    """A normal distribution parametrised by its natural parameter.

    Args:
        lam (column vector): Mean premultiplied by the precision.
        prec (matrix): Precision, the inverse covariance.
    """

    def __init__(self, lam, prec):
        self.lam = lam
        # The Cholesky of `self.prec` will be cached.
        self.prec = convert(prec, AbstractMatrix)

        self._mean = None
        self._var = None
        self._m2 = None

    @classmethod
    def from_normal(cls, dist):
        """Construct from a normal distribution.

        Args:
            dist (distribution): Normal distribution to construct from.

        Returns:
            :class:`.NaturalNormal`: Normal distribution parametrised by the natural
                parameters of `dist`.
        """
        return cls(B.cholsolve(B.chol(dist.var), dist.mean), B.pd_inv(dist.var))

    def to_normal(self):
        """Convert to normal distribution parametrised by a mean and variance.

        Returns:
            :class:`stheno.Normal`: Normal distribution parametrised by the a mean
                and variance.
        """
        return Normal(self.mean, self.var)

    @property
    def dtype(self):
        """dtype: Data type."""
        return B.dtype(self.lam, self.prec)

    @property
    def dim(self):
        """int: Dimensionality."""
        return B.shape_matrix(self.prec, 0)

    @property
    def mean(self):
        """column vector: Mean."""
        if self._mean is None:
            self._mean = B.cholsolve(B.chol(self.prec), self.lam)
        return self._mean

    @property
    def var(self):
        """matrix: Variance."""
        if self._var is None:
            self._var = B.pd_inv(self.prec)
        return self._var

    @property
    def m2(self):
        """matrix: Second moment."""
        if self._m2 is None:
            self._m2 = B.cholsolve(B.chol(self.prec), self.prec + B.outer(self.lam))
            self._m2 = B.cholsolve(B.chol(self.prec), B.transpose(self._m2))
        return self._m2

    def sample(self, state, num=1):
        """Sample.

        Args:
            state (random state): Random state.
            num (int): Number of samples.

        Returns:
            tuple[random state, tensor]: Random state and sample.
        """
        state, noise = Normal(self.prec).sample(state, num)
        sample = B.cholsolve(B.chol(self.prec), B.add(noise, self.lam))
        # Remove the matrix type if there is no structure. This eases working with
        # JITs, which aren't happy with matrix types.
        if not structured(sample):
            sample = B.dense(sample)
        return state, sample

    @_dispatch
    def kl(self, other: "NaturalNormal"):
        """Compute the Kullback-Leibler divergence with respect to another normal
        parametrised by its natural parameters.

        Args:
            other (:class:`.NaturalNormal`): Other.

        Returns:
            scalar: KL divergence with respect to `other`.
        """
        ratio = B.solve(B.chol(self.prec), B.chol(other.prec))
        diff = self.mean - other.mean
        return 0.5 * (
            B.sum(ratio ** 2)
            - B.logdet(B.mm(ratio, ratio, tr_a=True))
            + B.sum(B.mm(other.prec, diff) * diff)
            - B.cast(self.dtype, self.dim)
        )

    def logpdf(self, x):
        """Compute the log-pdf of some data.

        Args:
            x (column vector): Data to compute log-pdf of.

        Returns:
            scalar: Log-pdf of `x`.
        """
        diff = B.subtract(x, self.mean)
        return -0.5 * (
            -B.logdet(self.prec)
            + B.cast(self.dtype, self.dim) * B.cast(self.dtype, B.log_2_pi)
            + B.sum(B.mm(self.prec, diff) * diff)
        )
