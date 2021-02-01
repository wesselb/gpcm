import lab as B
import numpy as np
from numpy.testing import assert_allclose

__all__ = ["approx", "assert_positive_definite"]


def approx(x, y, atol=0, rtol=1e-8):
    """Assert that two tensors are approximately equal.

    Args:
        x (tensor): First tensor.
        y (tensor): Second tensor.
        atol (scalar, optional): Absolute tolerance. Defaults to `0`.
        rtol (scalar, optional): Relative tolerance. Defaults to `1e-8`.
    """

    assert_allclose(B.dense(x), B.dense(y), atol=atol, rtol=rtol)


def assert_positive_definite(x):
    """Assert that a matrix is positive definite by testing that its
    Cholesky decomposition computes.


    Args:
        x (matrix): Matrix that should be positive definite.
    """
    np.linalg.cholesky(B.to_numpy(x))
