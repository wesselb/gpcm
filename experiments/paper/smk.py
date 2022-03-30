import argparse

import lab as B
import matplotlib.pyplot as plt
import wbml.metric as metric
import wbml.out as out
from stheno import EQ, GP
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak, pdfcrop, tex

from gpcm import GPCM
from gpcm.util import estimate_psd

out.report_time = True
B.epsilon = 1e-8
wd = WorkingDirectory("server", "_experiments", "smk_comparison", observe=True)
tex()

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
args = parser.parse_args()

# Setup experiment.
noise = 0.1
t = B.linspace(0, 40, 200)
t_k = B.linspace(0, 4, 200)

# Setup GPCM models.
window = 2
scale = 0.25
n_u = 80
n_z = 80

# Sample data.
kernel = (lambda x: B.sin(B.pi * x)) * EQ() + (lambda x: B.cos(B.pi * x)) * EQ()
y = B.flatten(GP(kernel)(t, noise).sample())
k = B.flatten(kernel(t_k, 0))


def extract(pred):
    return (pred.x, pred.mean, pred.err_95_lower, pred.err_95_upper)


if args.train:
    model = GPCM(
        scheme="mean-field",
        window=window,
        scale=scale,
        noise=noise,
        n_u=n_u,
        n_z=n_z,
        t=t,
    )
    model.fit(t, y, iters=30_000)
    k_pred_mf = extract(model.condition(t, y).predict_kernel(t_k))
    psd_pred_mf = extract(model.condition(t, y).predict_psd())

    model = GPCM(
        scheme="structured",
        window=window,
        scale=scale,
        noise=noise,
        n_u=n_u,
        n_z=n_z,
        t=t,
    )
    model.fit(t, y, iters=30_000)
    k_pred_struc = extract(model.condition(t, y).predict_kernel(t_k))
    psd_pred_struc = extract(model.condition(t, y).predict_psd())

    wd.save((k_pred_mf, psd_pred_mf, k_pred_struc, psd_pred_struc), "preds.pickle")
else:
    k_pred_mf, psd_pred_mf, k_pred_struc, psd_pred_struc = wd.load("preds.pickle")


# Plot stuff.

plt.figure(figsize=(7.5, 3.75))
plt.subplot(1, 2, 1)
plt.plot(t_k, k, label="Truth", style="train")

t, mean, err_95_lower, err_95_upper = k_pred_struc
sig = (err_95_upper - err_95_lower) / (2 * 1.96)
var = sig ** 2

print("Struc")
inds = t <= 3
print(metric.mll(mean[inds], var[inds], k[inds]))
print(metric.rmse(mean[inds], k[inds]))

plt.plot(t, mean, label="Structured", style="pred")
plt.fill_between(
    t,
    err_95_lower,
    err_95_upper,
    style="pred",
)
plt.plot(t, err_95_upper, style="pred", lw=1)
plt.plot(t, err_95_lower, style="pred", lw=1)

t, mean, err_95_lower, err_95_upper = k_pred_mf
sig = (err_95_upper - err_95_lower) / (2 * 1.96)
var = sig ** 2

print("MF")
inds = t <= 3
print(metric.mll(mean[inds], var[inds], k[inds]))
print(metric.rmse(mean[inds], k[inds]))

plt.plot(t, mean, label="Mean-field", style="pred2")
plt.fill_between(
    t,
    err_95_lower,
    err_95_upper,
    style="pred2",
)
plt.plot(t, err_95_upper, style="pred2", lw=1)
plt.plot(t, err_95_lower, style="pred2", lw=1)

plt.xlabel("Time (s)")
plt.ylabel("Covariance")
plt.title("Kernel")

plt.xlim(0, 4)
plt.ylim(-0.75, 1.25)
plt.yticks([-0.5, 0, 0.5, 1])
tweak(legend=False)

plt.subplot(1, 2, 2)
t_k = B.linspace(-8, 8, 1001)
freqs, psd = estimate_psd(t, kernel(t, 0).flatten(), db=True)
inds = freqs <= 1
freqs = freqs[inds]
psd = psd[inds]
plt.plot(freqs, psd, label="Truth", style="train")

t, mean, err_95_lower, err_95_upper = psd_pred_struc
inds = t <= 1
t = t[inds]
mean = mean[inds]
err_95_lower = err_95_lower[inds]
err_95_upper = err_95_upper[inds]
plt.plot(t, mean, label="Structured", style="pred")
plt.fill_between(
    t,
    err_95_lower,
    err_95_upper,
    style="pred",
)
plt.plot(t, err_95_upper, style="pred", lw=1)
plt.plot(t, err_95_lower, style="pred", lw=1)

t, mean, err_95_lower, err_95_upper = psd_pred_mf
inds = t <= 1
t = t[inds]
mean = mean[inds]
err_95_lower = err_95_lower[inds]
err_95_upper = err_95_upper[inds]
plt.plot(t, mean, label="Mean-field", style="pred2")
plt.fill_between(
    t,
    err_95_lower,
    err_95_upper,
    style="pred2",
)
plt.plot(t, err_95_upper, style="pred2", lw=1)
plt.plot(t, err_95_lower, style="pred2", lw=1)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectral density (dB)")
plt.title("PSD")
plt.xlim(0, 1)
plt.ylim(-20, 10)
tweak(legend=True)
plt.savefig(wd.file("smk.pdf"))
pdfcrop(wd.file("smk.pdf"))
plt.show()
