import lab as B
import numpy as np
from wbml.data.snp import load

from gpcm.experiment import setup, run

B.epsilon = 1e-6   # This experiment requires a bit of help to not `NaN` out.

args, wd = setup("snp")

data = load()
data = data.iloc[-300:]
t = np.array(data.index)
y = np.array(data["volume"])

# Convert to days since start.
t = (t - t[0]) * 365

# Normalise to zero mean and unity variance.
y -= B.mean(y)
y /= B.std(y)

# Setup GPCM models.
noise = 0.05
window = 7
scale = 0.1

run(
    args=args,
    wd=wd,
    noise=noise,
    window=window,
    scale=scale,
    t=t,
    y=y,
    n_u=100,
    n_z=300,
    x_range={"psd": (0, 0.5)},
    y_range={"psd": (-20, 20)},
)
