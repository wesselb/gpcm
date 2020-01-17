import lab as B
import numpy as np
import pytest

import gpcm.gpcm as gpcm
import gpcm.gprv as gprv


@pytest.mark.parametrize('Model',
                         [lambda **kw_args: gpcm.GPCM(**kw_args),
                          lambda **kw_args: gpcm.GPCM(causal=False, **kw_args),
                          lambda **kw_args: gprv.GPRV(gamma=1, **kw_args)])
def test_prior_power(Model):
    t = B.linspace(0, 10, 50)
    t_u = B.zeros(1)
    model = Model(window=2, scale=1, n_u=10, t=t)
    K_u = model.compute_K_u()

    # Estimate power with Monte Carlo.
    powers = []
    for _ in range(5_000):
        u = B.sample(K_u)[:, 0]
        powers.append(model.kernel_approx(t_u, t_u, u)[0, 0])
    power = np.mean(powers)

    assert B.abs(power - 1) < 5e-2
