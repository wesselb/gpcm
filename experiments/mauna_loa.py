import lab as B
import numpy as np
from experiments.experiment import run, setup
from wbml.data.mauna_loa import load

args, wd = setup("mauna_loa")

n = 200
data = load(detrend_method="gp")
t = np.array(data.index)[-n:]
t = t - t[0]
y = np.array(data["ppm_detrended"])[-n:]

# Normalise to zero mean and unity variance.
y -= B.mean(y)
y /= B.std(y)

# Setup GPCM models.
noise = 0.05
window = 5
scale = 1 / 12

run(
    args=args,
    wd=wd,
    noise=noise,
    window=window,
    scale=scale,
    fix_window_scale=True,
    t=t,
    y=y,
    n_z=n,
    n_u=100,
    y_range={"kernel": (-0.5, 2), "psd": (-30, 20)},
)
