import lab as B
import numpy as np
import pytest

from gpcm import GPCM, CGPCM, GPRV
from .util import approx


@pytest.mark.parametrize(
    "Model",
    [
        lambda **kw_args: GPCM(**kw_args),
        lambda **kw_args: CGPCM(**kw_args),
        lambda **kw_args: GPRV(gamma=1, **kw_args),
    ],
)
def test_prior_power(Model):
    t_u = B.zeros(1)
    model = Model(window=2, scale=1, n_u=10, t=(0, 10))
    K_u = model.compute_K_u()

    # Estimate power with Monte Carlo.
    powers = []
    for _ in range(2_000):
        u = B.sample(K_u)[:, 0]
        powers.append(model.kernel_approx(t_u, t_u, u)[0, 0])
    power = np.mean(powers)

    approx(power, 1, atol=5e-2)
