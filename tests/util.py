from numpy.testing import assert_allclose, assert_array_almost_equal
import numpy as np
import lab as B

__all__ = ["allclose", "approx", "assert_positive_definite"]

allclose = assert_allclose
approx = assert_array_almost_equal


def assert_positive_definite(x):
    """Assert that a matrix is positive definite by testing that its
    Cholesky decomposition computes.


    Args:
        x (matrix): Matrix that should be positive definite.
    """
    np.linalg.cholesky(B.to_numpy(x))
