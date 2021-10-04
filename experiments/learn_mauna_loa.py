import lab as B
import numpy as np
from gpcm.experiment import run, setup
from wbml.data.mauna_loa import load

args, wd = setup("mauna_loa")

n = 300
data = load()
t = np.array(data.index)[-n:]
y = np.array(data["ppm_detrended"])[-n:]

t = t - t[0]  # Why does this help the numerics? Avoid cancellations?

# Normalise to zero mean and unity variance.
y -= B.mean(y)
y /= B.std(y)

# Setup GPCM models.
noise = 0.05
window = min(max(t), 15)
scale = 3 / 12

run(
    args=args,
    wd=wd,
    noise=noise,
    window=window,
    scale=scale,
    t=t,
    y=y,
    y_range={"kernel": (-0.5, 2), "psd": (-30, 20)},
)
