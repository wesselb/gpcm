import gpcm.gpcm as gpcm

from .util import allclose


def test_scale_factor_conversion():
    allclose(gpcm.scale_to_factor(gpcm.factor_to_scale(1)), 1)
    allclose(gpcm.factor_to_scale(gpcm.scale_to_factor(1)), 1)
