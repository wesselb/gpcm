import lab as B
import numpy as np
from gpcm.experiment import setup, run
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
noise = 0.1
window = 15
scale = 2 / 12
n_u = 6 * (window * 2)   # Roughly six inducing points per year
n_z = n // 2

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
