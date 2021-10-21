import argparse
import datetime

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wbml.out as out
from scipy.signal import periodogram
from scipy.stats import linregress
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak, pdfcrop

from gpcm import GPRVM

# Setup experiment.
out.report_time = True
B.epsilon = 1e-6

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--predict", action="store_true")
parser.add_argument("--server", action="store_true")
args = parser.parse_args()

wd = WorkingDirectory("server", "_experiments", "vix_analyse", observe=True)

# Load and process data.
data = pd.read_parquet("experiments/vix.parquet")
data = data.set_index("date")
lower = datetime.datetime(2000, 1, 1)
upper = datetime.datetime(2001, 1, 1)
data = data[(data.index >= lower) & (data.index < upper)]
# Convert to days since start. The data type is a timestamp in ns.
t = np.array(data.index - data.index[0], dtype=float) / 1e9 / 3600 / 24
y = np.log(np.array(data.open)).flatten()

# Normalise.
y_scale = y.std()
y = (y - y.mean()) / y.std()

# Configure GPCM models.
window = 7 * 6
scale = 3
n_u = 60
n_z = len(t)

# Setup, fit, and save models.
model = GPRVM(
    window=window,
    scale=scale,
    noise=0.05,
    n_u=n_u,
    n_z=n_z,
    t=t,
)
if args.train:
    model.fit(t, y, iters=50_000, rate=2e-2, optimise_hypers=20_000)
    model.save(wd.file("model.pickle"))
else:
    model.load(wd.file("model.pickle"))

# Make and save predictions.
if args.predict:
    posterior = model.condition(t, y)
    pred_f = (t,) + posterior.predict(t)
    pred_psd = posterior.predict_psd()
    pred_psd = (
        pred_psd.x,
        pred_psd.mean,
        pred_psd.err_95_lower,
        pred_psd.err_95_upper,
        pred_psd.all_samples,
    )
    pred_k = posterior.predict_kernel()
    pred_k = (pred_k.x, pred_k.mean, pred_k.var)
    wd.save(pred_f, "pred_f.pickle")
    wd.save(pred_psd, "pred_psd.pickle")
    wd.save(pred_k, "pred_k.pickle")
else:
    pred_f = wd.load("pred_f.pickle")
    pred_psd = wd.load("pred_psd.pickle")
    pred_k = wd.load("pred_k.pickle")


def estimate_H(freqs, psd):
    inds = (0.15 <= freqs) & (freqs <= 0.5)

    slope, intercept = linregress(
        10 * np.log10(2 * np.pi * freqs[inds]),
        psd[inds],
    )[:2]
    c = 10 ** (intercept / 10)
    H = 0.5 * (1 - slope)
    return c, H


freqs, mean, lower, upper, samps = pred_psd
upper_freq = 0.5
samps = samps[freqs <= upper_freq, :]
mean = mean[freqs <= upper_freq]
lower = lower[freqs <= upper_freq]
upper = upper[freqs <= upper_freq]
freqs = freqs[freqs <= upper_freq]

instance = model()
spec_x = (2 * instance.lam) / (instance.lam ** 2 + (2 * B.pi * freqs) ** 2)
spec_x *= instance.alpha_t ** 2 / (2 * instance.alpha)
spec_x = 10 * B.log(spec_x) / B.log(10)

plt.figure(figsize=(12, 3.5))

plt.subplot(1, 2, 1)
plt.title("PSD")
plt.plot(
    freqs,
    mean,
    label="$|\\mathcal{F}h(f)|^2\\mathcal{F}k_{x}(f)$",
    style="pred",
    zorder=1,
)
plt.fill_between(freqs, lower, upper, style="pred", zorder=1)
plt.plot(freqs, lower, style="pred", lw=1, zorder=1)
plt.plot(freqs, upper, style="pred", lw=1, zorder=1)
per_x, per = periodogram(y)
plt.plot(per_x, 10 * np.log10(per), label="Periodogram", lw=1, style="train", zorder=0)
plt.xlim(0, 0.5)
plt.ylim(-30, 20)
plt.xlabel("Frequency $f$ (day${}^{-1}$)")
plt.ylabel("Spectral density (dB)")
tweak(legend_loc="upper right")

mean -= spec_x
lower -= spec_x
upper -= spec_x

mean -= 7.5
lower -= 7.5
upper -= 7.5
spec_x += 7.5

plt.subplot(1, 2, 2)
plt.title("Decomposition of PSD")
plt.scatter(0.2, -7.5, c="k", s=15, marker="o")
plt.plot([0.20001, 0.2, 1], [-100, -7.5, -7.5], lw=1, c="k")
plt.plot(freqs, mean, label="$|\\mathcal{F}h(f)|^2$", style="pred")
plt.plot(freqs, spec_x, style="pred2", label="$\\mathcal{F}k_{x}(f)$")
plt.fill_between(freqs, lower, upper, style="pred")
plt.plot(freqs, lower, style="pred", lw=1)
plt.plot(freqs, upper, style="pred", lw=1)
plt.ylim(-25, 15)
plt.xlim(0, 0.5)
plt.xlabel("Frequency $f$ (day${}^{-1}$)")
tweak(legend_loc="upper right")

plt.savefig(wd.file("psd.pdf"))
pdfcrop(wd.file("psd.pdf"))

plt.show()
