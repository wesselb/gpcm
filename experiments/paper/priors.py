import argparse

import lab as B
import matplotlib.pyplot as plt
from wbml.experiment import WorkingDirectory
from wbml.plot import tex, tweak, pdfcrop

from gpcm import GPCM, CGPCM, RGPCM

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
args = parser.parse_args()

# Setup script.
B.epsilon = 1e-10
tex()
wd = WorkingDirectory("_experiments", "priors", seed=0)

# Construct models.
models = [
    GPCM(window=2, scale=0.5, n_u=30, t=(0, 10)),
    CGPCM(window=2, scale=0.5, n_u=30, t=(0, 10)),
    RGPCM(window=2, scale=0.5, n_u=30, t=(0, 10)),
]

# Instantiate models.
models = [model() for model in models]


def _extract_samples(quantities):
    x = quantities.x
    samples = quantities.all_samples
    return (
        B.concat(-x[::-1][:-1], x),
        B.concat(samples[::-1, :][:-1, :], samples, axis=0),
    )


# Perform sampling.
if args.train:
    ks = [_extract_samples(model.predict_kernel(num_samples=20000)) for model in models]
    psds = [_extract_samples(model.predict_psd(num_samples=20000)) for model in models]
    model_ks, model_psds = ks, psds
    wd.save((model_ks, model_psds), "samples.pickle")
else:
    model_ks, model_psds = wd.load("samples.pickle")

# Plot.
plt.figure(figsize=(15, 2.5))

for i, (model, (x, ks)) in enumerate(zip(models, model_ks)):
    plt.subplot(1, 6, 1 + i)
    for q in [1, 5, 10, 20, 30, 40]:
        plt.fill_between(
            x,
            B.quantile(ks, q / 100, axis=1),
            B.quantile(ks, 1 - q / 100, axis=1),
            facecolor="tab:blue",
            alpha=0.2,
        )
    plt.plot(x, B.mean(ks, axis=1), c="black")
    if hasattr(model, "t_u"):
        plt.scatter(model.t_u, model.t_u * 0, s=5, marker="o", c="black")
    plt.title(model.name + " (Kernel)")
    plt.xlabel("Time (s)")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-0.5, 1.25)
    wbml.plot.tweak(legend=False)

for i, (model, (freqs, psds)) in enumerate(zip(models, model_psds)):
    plt.subplot(1, 6, 4 + i)

    def apply_to_psd(f):
        raw = 10 ** (psds / 10)
        return 10 * B.log(f(raw)) / B.log(10)

    for q in [1, 5, 10, 20, 30, 40]:
        plt.fill_between(
            freqs,
            apply_to_psd(lambda x: B.quantile(x, q / 100, axis=1)),
            apply_to_psd(lambda x: B.quantile(x, 1 - q / 100, axis=1)),
            facecolor="tab:blue",
            alpha=0.2,
        )
    # Careful: take the mean in PSD space!
    plt.plot(
        freqs,
        apply_to_psd(lambda x: B.mean(x, axis=1)),
        c="black",
    )
    plt.title(model.name + " (PSD)")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(-3, 3)
    plt.ylim(-30, 5)
    tweak(legend=False)

plt.tight_layout()
plt.savefig(wd.file("priors.pdf"))
pdfcrop(wd.file("priors.pdf"))
plt.show()
