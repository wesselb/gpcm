import numpy as np
import torch
from wbml.data.mauna_loa import load
import lab.torch as B
from sklearn.linear_model import LinearRegression

from gpcm.experiment import setup, run

args, wd = setup("mauna_loa")

data = load()
data = data[(2000 <= data.index) & (data.index < 2015)]

t = torch.tensor(np.array(data.index))
y = torch.tensor(np.array(data["ppm_detrended"]))

# Detrend once more, because we have selected a subset.
lr = LinearRegression()
lr.fit(t[:, None], y)
y = y - lr.predict(t[:, None])

t = t - t[0]  # Why does this help the numerics? Avoid cancellations?

# Normalise to zero mean and unity variance.
y -= B.mean(y)
y /= B.std(y)

# Setup GPCM models.
noise = 0.01
window = 4
scale = 0.1
n_u = 100
n_z = 100

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
)
