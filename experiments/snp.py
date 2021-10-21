import argparse

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.out as out
from scipy.stats import linregress
from wbml.data.snp import load
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak, pdfcrop

from gpcm import GPRVM
from gpcm.util import autocorr

# Setup experiment.
out.report_time = True
B.epsilon = 1e-8

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--predict", action="store_true")
parser.add_argument("--server", action="store_true")
args = parser.parse_args()

if args.server:
    wd = WorkingDirectory("server", "_experiments", "snp", observe=True)
else:
    wd = WorkingDirectory("_experiments", "snp")

# Load and process data.
n = 500
data = load()
t = np.array(data.index)[-n:]
t = (t - t[0]) * 365
y = np.log(np.array(data)[-n:, 0])

# Normalise.
y = (y - y.mean()) / y.std()

# Configure GPCM models.
window = 30
scale = 3
n_u = 50
n_z = 200

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
    model.fit(t, y, iters=50_000, rate=2e-2, optimise_hypers=10_000)
    model.save(wd.file("model.pickle"))
else:
    model.load(wd.file("model.pickle"))

# Make and save predictions.
if args.predict:
    posterior = model.condition(t, y)
    pred_f = (t,) + posterior.predict(t)
    pred_psd = posterior.predict_kernel()
    pred_psd = (pred_psd.x, pred_psd.mean, pred_psd.var)
    wd.save(pred_f, "pred_f.pickle")
    wd.save(pred_psd, "pred_psd.pickle")
else:
    pred_f = wd.load("pred_f.pickle")
    pred_psd = wd.load("pred_psd.pickle")
    pred_k = wd.load("pred_k.pickle")


plt.figure(figsize=(12, 3.5))

plt.subplot(1, 2, 1)
plt.title("Kernel")
freqs, mean, var = pred_k
err = 1.96 * B.sqrt(var)
mean *= scale ** 2
err *= scale ** 2
var_mean = mean[0] + model().noise * scale ** 2
var_err = err[0]
plt.plot(freqs, mean, label="Kernel", style="pred", zorder=0)
plt.fill_between(freqs, mean - err, mean + err, style="pred", zorder=0)
plt.plot(freqs, mean - err, style="pred", lw=1, zorder=0)
plt.plot(freqs, mean + err, style="pred", lw=1, zorder=0)

ones = B.ones(freqs)
mean = var_mean * ones
err = var_err * ones
plt.plot(freqs, mean, style="pred2", zorder=0, label="Variance")
plt.fill_between(freqs, mean - err, mean + err, style="pred2", zorder=0, alpha=0.15)
plt.plot(freqs, mean - err, style="pred2", lw=1, zorder=0)
plt.plot(freqs, mean + err, style="pred2", lw=1, zorder=0)
plt.plot(
    t - t[0],
    autocorr(y * scale, cov=True, lags=len(t)),
    style="train",
    label="Empirical",
    zorder=0,
    lw=1,
)
plt.xlim(0, 60)
plt.ylim(-0.05, 0.25)
plt.xlabel("Time (days)")
plt.ylabel("Covariance (USD${}^{2}$)")
tweak(legend_loc="center right")


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

samps = samps[freqs <= 0.5, :]
mean = mean[freqs <= 0.5]
lower = lower[freqs <= 0.5]
upper = upper[freqs <= 0.5]
freqs = freqs[freqs <= 0.5]
err = 1.96 * B.sqrt(var)

instance = model()
spec_x = (2 * instance.lam) / (instance.lam ** 2 + (2 * B.pi * freqs) ** 2)
spec_x *= instance.alpha_t ** 2 / (2 * instance.alpha)
spec_x = 10 * B.log(spec_x) / B.log(10)

c, H = estimate_H(freqs, mean - spec_x)
print("H:", H)
rough_spec = c * (2 * np.pi * freqs) ** (1 - 2 * H)
rough_spec = 10 * B.log(rough_spec) / B.log(10)

mean -= spec_x
lower -= spec_x
upper -= spec_x

plt.subplot(1, 2, 2)
plt.title("PSD")
plt.plot(freqs, mean, label="$|g(2\pi f)|^2$", style="pred")
plt.plot(freqs, spec_x, style="pred2", label="$\phi_x(2\pi f)$")
plt.fill_between(freqs, lower, upper, style="pred")
plt.plot(freqs, lower, style="pred", lw=1)
plt.plot(freqs, upper, style="pred", lw=1)
plt.plot(
    freqs,
    rough_spec,
    c="k",
    label=f"${c:.2f} \cdot |2\pi f|^{{1 - 2 \cdot {H:.3f}}}$",
    lw=1,
)
plt.xlim(0, 0.5)
plt.xlabel("Frequency $f$ (Hz)")
plt.ylabel("Spectral density (dB)")
tweak(legend_loc="lower left")
plt.savefig(wd.file("psd.pdf"))
pdfcrop(wd.file("psd.pdf"))

plt.show()
