import lab as B
import numpy as np
from gpcm.experiment import setup, run
from sklearn.linear_model import LinearRegression
from wbml.data.mauna_loa import load

args, wd = setup("mauna_loa")

data = load()
t = np.array(data.index)
y = np.array(data["ppm_detrended"])

# Detrend once more, because we have selected a subset.
lr = LinearRegression()
lr.fit(t[:, None], y)
y = y - lr.predict(t[:, None])

t = t - t[0]  # Why does this help the numerics? Avoid cancellations?

# Normalise to zero mean and unity variance.
y -= B.mean(y)
y /= B.std(y)

# Setup GPCM models.
noise = 0.1
window = 4
scale = 0.2
n_u = 50
n_z = 300

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
    y_range={"kernel": (-0.5, 2), "psd": (-30, 10)},
)
