import lab as B
import numpy as np
from gpcm.experiment import run, setup
from wbml.data.mauna_loa import load

args, wd = setup("mauna_loa")

n = 400
data = load()
t = np.array(data.index)[-n:]
y = np.array(data["ppm_detrended"])[-n:]

t = t - t[0]  # Why does this help the numerics? Avoid cancellations?

# Normalise to zero mean and unity variance.
y -= B.mean(y)
y /= B.std(y)

# Setup GPCM models.
noise = 0.1
window = min(max(t) / 2, 15)
scale = 2 / 12
# Set two points per wiggle of the filter.
n_u = int(2 * (2 * window) / scale)
# Set two inducing points per frequency and add Nyquist correction.
n_z = int(2.2 * 2 * 2 * max(t))

run(
    args=args,
    wd=wd,
    noise=noise,
    window=window,
    scale=scale,
    t=t,
    y=y,
    n_u=n_u,
    n_z=n_z,
    y_range={"kernel": (-0.5, 2), "psd": (-30, 20)},
)
