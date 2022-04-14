import argparse

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.out
from stheno import GP
from wbml.experiment import WorkingDirectory
from wbml.plot import tex, tweak, pdfcrop

from gpcm import CGPCM
from gpcm.util import closest_psd

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
args = parser.parse_args()

# Setup script.
B.epsilon = 1e-10
tex()
wd = WorkingDirectory("_experiments", "sample_interpolate", seed=14)


def sample(model, t, noise_f):
    """Sample from a model.

    Args:
        model (:class:`gpcm.model.AbstractGPCM`): Model to sample from.
        t (vector): Time points to sample at.
        noise_f (vector): Noise for the sample of the function. Should have the
            same size as `t`.

    Returns:
        tuple[vector, ...]: Tuple containing kernel samples, filter samples, and
            function samples.
    """
    ks, us, fs = [], [], []

    # In the below, we look at the third inducing point, because that is the one
    # determining the value of the filter at zero: the CGPCM adds two extra inducing
    # points to the left.

    # Get a smooth sample.
    u1 = B.ones(model.n_u)
    while B.abs(u1[2]) > 1e-2:
        u1 = B.sample(model.compute_K_u())[:, 0]
    u = GP(model.k_h())
    u = u | (u(model.t_u), u1)
    u1_full = u(t).mean.flatten()

    # Get a rough sample.
    u2 = B.zeros(model.n_u)
    while u2[2] < 0.5:
        u2 = B.sample(model.compute_K_u())[:, 0]
    u = GP(model.k_h())
    u = u | (u(model.t_u), u2)
    u2_full = u(t).mean.flatten()

    with wbml.out.Progress(name="Sampling", total=5) as progress:
        for c in [0, 0.1, 0.23, 0.33, 0.5]:
            # Sample kernel.
            K = model.kernel_approx(t, t, c * u2 + (1 - c) * u1)
            wbml.out.kv("Sampled variance", K[0, 0])
            K = K / K[0, 0]
            ks.append(K[0, :])

            # Store filter.
            us.append(c * u2_full + (1 - c) * u1_full)

            # Sample function.
            f = B.matmul(B.chol(closest_psd(K)), noise_f)
            fs.append(f)

            progress()

    return ks, us, fs


t = np.linspace(0, 10, 300)
noise_f = np.random.randn(len(t), 1)

# Construct model.
model = CGPCM(window=2, scale=1, n_u=10, t=t)

# Instantiate model.
models = model()

# Perform sampling.
if args.train:
    ks, us, fs = sample(model, t, noise_f)
    wd.save((ks, us, fs), "samples.pickle")
else:
    ks, us, fs = wd.load("samples.pickle")

# Plot.
plt.figure(figsize=(15, 4))

for i, (k, u, f) in enumerate(zip(ks, us, fs)):
    plt.subplot(3, 5, 1 + i)
    plt.plot(
        B.concat(-t[::-1][:-1], t),
        B.concat(u[:-1] * 0, u),
        lw=1,
    )
    if hasattr(model, "t_u"):
        plt.scatter(model.t_u, model.t_u * 0, s=5, marker="o", c="black")
    # plt.xlabel("Time (s)")
    if i == 0:
        plt.ylabel("$h$")
    plt.xlim(-6, 6)
    plt.ylim(-0.5, 1)
    tweak(legend=False)

    plt.subplot(3, 5, 6 + i)
    plt.plot(
        B.concat(-t[::-1][:-1], t),
        B.concat(k[::-1][:-1], k),
        lw=1,
    )
    if hasattr(model, "t_u"):
        plt.scatter(model.t_u, model.t_u * 0, s=5, marker="o", c="black")
    # plt.xlabel("Time (s)")
    if i == 0:
        plt.ylabel("$k_{f\,|\,h}$")
    plt.xlim(-6, 6)
    plt.ylim(-0.35, 1.2)
    tweak(legend=False)

    plt.subplot(3, 5, 11 + i)
    plt.plot(t, f, lw=1)
    if hasattr(model, "t_z"):
        plt.scatter(model.t_z, model.t_z * 0, s=5, marker="o", c="black")
    # plt.xlabel("Time (s)")
    if i == 0:
        plt.ylabel("$f$")
    plt.xlim(0, 8)
    tweak(legend=False)

plt.tight_layout()
plt.savefig(wd.file("interpolation.pdf"))
pdfcrop(wd.file("interpolation.pdf"))
plt.show()
