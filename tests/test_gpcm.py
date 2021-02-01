import gpcm.gpcm as gpcm

from .util import approx


def test_scale_factor_conversion():
    approx(gpcm.scale_to_factor(gpcm.factor_to_scale(1)), 1)
    approx(gpcm.factor_to_scale(gpcm.scale_to_factor(1)), 1)
