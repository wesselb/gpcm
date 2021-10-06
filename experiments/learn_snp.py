import lab as B
import numpy as np
from wbml.data.snp import load

from gpcm.experiment import setup, run

args, wd = setup("snp")

n = 200
data = load()
data = data.iloc[-n:]
t = np.array(data.index)
y = np.array(data["volume"])

t = t - t[0]  # Why does this help the numerics? Avoid cancellations?

# Normalise to zero mean and unity variance.
y -= B.mean(y)
y /= B.std(y)

# Setup GPCM models.
noise = 0.05
window = 7 / 365
scale = 0.5 / 365

run(
    args=args,
    wd=wd,
    noise=noise,
    window=window,
    scale=scale,
    t=t,
    y=y,
    n_u=150,
    n_z=n,
    x_range={"psd": (0, 500)},
    y_range={"psd": (-60, 0)},
)
