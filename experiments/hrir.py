import argparse

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as sio_wav
import wbml.out as out
from gpcm import GPCM, CGPCM
from gpcm.util import min_phase
from slugify import slugify
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

# Setup experiment.
out.report_time = True
B.epsilon = 1e-6

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--predict", action="store_true")
parser.add_argument("--server", action="store_true")
args = parser.parse_args()

if args.server:
    wd = WorkingDirectory("server", "_experiments", "hrir3", observe=True)
else:
    wd = WorkingDirectory("_experiments", "hrir3")

# Load and process data.
fs, h = sio_wav.read("experiments/R20e280a.wav")
t_h = np.arange(len(h)) / float(fs)
h = h[t_h <= 4e-3]  # Get rid of noisy tail
t_h = t_h[t_h <= 4e-3] * 1000

# Generate data.
n = 300
x = np.random.randn(n + 200)
y = np.convolve(h, x)[100 : 100 + n]
y = y / np.std(y)  # Normalise to unity variance
t = (t_h[1] - t_h[0]) * np.arange(len(y))

# Configure GPCM models.
window = 1.0
scale = 0.05
n_u = 100
n_z = 300


def model_path(model):
    return (slugify(model.name), slugify(model.scheme))


# Setup, fit, and save models.
models = [
    Model(
        window=window,
        scale=scale,
        noise=0.1,
        n_u=n_u,
        n_z=n_z,
        t=t,
    )
    for Model in [CGPCM, GPCM]
]
if args.train:
    for model in models:
        model.fit(t, y, iters=200_000, rate=5e-2, optimise_hypers=500, num_samples=10)
        model.save(wd.file(*model_path(model), "model.pickle"))
else:
    for model in models:
        model.load(wd.file(*model_path(model), "model.pickle"))

# Make and save predictions.
preds_f = []
preds_k = []
preds_h = []
if args.predict:
    for model in models:
        # Perform predictions.
        posterior = model.condition(t, y)
        # pred_f = (t,) + posterior.predict(t)
        # pred_k = posterior.predict_kernel()
        # pred_k = (pred_k.x, pred_k.mean, pred_k.var)
        pred_h = posterior.predict_filter(min_phase=True)
        pred_h = (pred_h.x, pred_h.mean, pred_h.var)
        # Save predictions.
        # preds_f.append(pred_f)
        # preds_k.append(pred_k)
        preds_h.append(pred_h)
        # wd.save(pred_f, *model_path(model), "pred_f.pickle")
        # wd.save(pred_k, *model_path(model), "pred_k.pickle")
        wd.save(pred_h, *model_path(model), "pred_h.pickle")
else:
    for model in models:
        # preds_f.append(wd.load(*model_path(model), "pred_f.pickle"))
        # preds_k.append(wd.load(*model_path(model), "pred_k.pickle"))
        preds_h.append(wd.load(*model_path(model), "pred_h.pickle"))


h = min_phase(h)

plt.figure()
plt.subplot(1, 2, 1)
energy = np.trapz(h ** 2, t_h)
h = h / energy ** 0.5
plt.plot(t_h, h, label="HRIR", style="train")

t, mean, var = preds_h[0]
err = 1.96 * B.sqrt(var)
energy = np.trapz(mean ** 2, t)
err /= energy ** 0.5
mean /= energy ** 0.5

model = models[0]()
plt.scatter(model.t_u, model.t_u * 0, c="k", s=5)
plt.plot(t, mean, label=models[0].name, style="pred")
plt.fill_between(t, mean - err, mean + err, style="pred")
plt.plot(t, mean - err, style="pred", lw=1)
plt.plot(t, mean + err, style="pred", lw=1)

plt.xlim(0, window * 2)
plt.xlabel("Time (ms)")
plt.ylabel("Filter (normalised)")

tweak()
plt.subplot(1, 2, 2)
plt.plot(t_h, h, label="HRIR", style="train")

t, mean, var = preds_h[1]
err = 1.96 * B.sqrt(var)
energy = np.trapz(mean ** 2, t)
err /= energy ** 0.5
mean /= energy ** 0.5

plt.plot(t, mean, label=models[1].name, style="pred2")
plt.fill_between(t, mean - err, mean + err, style="pred2")
plt.plot(t, mean - err, style="pred2", lw=1)
plt.plot(t, mean + err, style="pred2", lw=1)

plt.xlim(0, window * 2)
plt.xlabel("Time (ms)")
plt.ylabel("Filter (normalised)")
tweak()

plt.show()
