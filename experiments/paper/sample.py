import argparse

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.out
from wbml.experiment import WorkingDirectory
from wbml.plot import tex, tweak, pdfcrop

from gpcm import GPCM, CGPCM, RGPCM
from gpcm.util import estimate_psd, closest_psd

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
args = parser.parse_args()

# Setup script.
B.epsilon = 1e-10
tex()
wd = WorkingDirectory("_experiments", "sample", seed=17)


def sample(model, t, noise_f):
    """Sample from a model.

    Args:
        model (:class:`gpcm.model.AbstractGPCM`): Model to sample from.
        t (vector): Time points to sample at.
        noise_f (vector): Noise for the sample of the function. Should have the
            same size as `t`.

    Returns:
        tuple[vector]: Tuple containing kernel samples and function samples.
    """
    ks, fs = [], []

    with wbml.out.Progress(name="Sampling", total=5) as progress:
        for i in range(5):
            # Sample kernel.
            u = B.sample(model.compute_K_u())[:, 0]
            K = model.kernel_approx(t, t, u)
            wbml.out.kv("Sampled variance", K[0, 0])
            K = K / K[0, 0]
            ks.append(K[0, :])

            # Sample function.
            f = B.matmul(B.chol(closest_psd(K)), noise_f)
            fs.append(f)

            progress()

    return ks, fs


t = np.linspace(0, 10, 400)
noise_f = np.random.randn(len(t), 1)

# Construct models.
models = [
    GPCM(window=2, scale=0.5, n_u=30, t=t),
    CGPCM(window=2, scale=0.5, n_u=30, t=t),
    RGPCM(window=2, scale=0.5, n_u=30, t=t),
]

# Instantiate models.
models = [model() for model in models]

# Perform sampling.
if args.train:
    model_ks, model_fs = zip(*[sample(model, t, noise_f) for model in models])
    wd.save((model_ks, model_fs), "samples.pickle")
else:
    model_ks, model_fs = wd.load("samples.pickle")

# Plot.
plt.figure(figsize=(15, 6))

for i, (model, ks, fs) in enumerate(zip(models, model_ks, model_fs)):
    plt.subplot(3, 2, 1 + 2 * i)
    plt.plot(t, np.stack(fs).T, lw=1)
    if hasattr(model, "t_z"):
        plt.scatter(model.t_z, model.t_z * 0, s=5, marker="o", c="black")
    plt.title(model.name)
    if i == 2:
        plt.xlabel("Time (s)")
    plt.xlim(0, 8)
    tweak(legend=False)

    plt.subplot(3, 4, 3 + 4 * i)
    plt.plot(t, np.stack(ks).T, lw=1)
    plt.scatter(model.t_u, model.t_u * 0, s=5, marker="o", c="black")
    plt.title("Kernel")
    if i == 2:
        plt.xlabel("Lag (s)")
    plt.xlim(0, 6)
    tweak(legend=False)

    # Estimate PSD.
    freqs, psds = zip(*[estimate_psd(t, k, db=True) for k in ks])
    freqs = freqs[0]
    psds = np.stack(psds).T

    plt.subplot(3, 4, 4 + 4 * i)
    plt.title("PSD (dB)")
    inds = np.arange(int(len(freqs) / 2))
    inds = inds[freqs[inds] <= 2]
    plt.plot(freqs[inds], psds[inds, :], lw=1)
    if i == 2:
        plt.xlabel("Frequency (Hz)")
    plt.xlim(0, 2)
    plt.ylim(-40, 10)
    tweak(legend=False)

plt.tight_layout()
plt.savefig(wd.file("sample.pdf"))
pdfcrop(wd.file("sample.pdf"))
plt.show()
