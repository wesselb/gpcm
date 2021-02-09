import numpy as np
import torch
from wbml.data.snp import load
import lab.torch as B

from gpcm.experiment import setup, run

args, wd = setup("snp")

data = load()
data = data.iloc[-200:]

t = torch.tensor(np.array(data.index))
y = torch.tensor(np.array(data["volume"]))

t = t - t[0]  # Why does this help the numerics? Avoid cancellations?

# Normalise to zero mean and unity variance.
y -= B.mean(y)
y /= B.std(y)

# Setup GPCM models.
noise = 0.05
window = 7 / 365
scale = 0.5 / 365
n_u = 50
n_z = 150

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
    x_range={"psd": (0, 500)},
    y_range={"psd": (-60, 0)}
)
