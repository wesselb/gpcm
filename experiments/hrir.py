import argparse

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.metric as metric
import wbml.out as out
from gpcm import GPCM
from gpcm.util import min_phase
from slugify import slugify
from wbml.data.kemar import load
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak


# Setup experiment.
out.report_time = True
B.epsilon = 1e-8

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--predict", action="store_true")
parser.add_argument("--server", action="store_true")
args = parser.parse_args()

if args.server:
    wd = WorkingDirectory("server", "_experiments", "hrir", observe=True)
else:
    wd = WorkingDirectory("_experiments", "hrir")

# Load and process data.
data = load()["right"][45, 45]
t_h = np.array(data.index)
h = min_phase(np.array(data))

# Generate data.
n = 300
x = np.random.randn(n + 200)
y = np.convolve(h, x)[100 : 100 + n]
y = y / np.std(y)  # Normalise to unity variance
t = (t_h[1] - t_h[0]) * np.arange(len(y))

# Configure GPCM models.
window = 0.1e-3
scale = 0.01e-3
n_u = 100
n_z = 150


def model_path(model):
    return (slugify(model.name), slugify(model.scheme))


# Setup, fit, and save models.
models = [
    Model(
        window=window,
        scale=scale,
        noise=0.05,
        n_u=n_u,
        n_z=n_z,
        t=t,
    )
    for Model in [GPCM]  # , CGPCM]
]
if args.train:
    for model in models:
        model.fit(t, y, iters=5000)
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
        pred_f = (t,) + posterior.predict(t)
        pred_k = posterior.predict_kernel()
        pred_k = (pred_k.x, pred_k.mean, pred_k.var)
        pred_h = posterior.predict_filter()
        pred_h = (pred_h.x, pred_h.mean, pred_h.var)
        # Save predictions.
        preds_f.append(pred_f)
        preds_k.append(pred_k)
        preds_h.append(pred_h)
        wd.save(pred_f, *model_path(model), "pred_f.pickle")
        wd.save(pred_k, *model_path(model), "pred_k.pickle")
        wd.save(pred_h, *model_path(model), "pred_h.pickle")
else:
    for model in models:
        preds_f.append(wd.load(*model_path(model), "pred_f.pickle"))
        preds_k.append(wd.load(*model_path(model), "pred_k.pickle"))
        preds_h.append(wd.load(*model_path(model), "pred_h.pickle"))



# Plot result.
plt.figure(figsize=(12, 3))
plt.scatter(t_train, normaliser.untransform(y_train), style="train", label="Train")
plt.scatter(t_test, y_test, style="test", label="Test")

plt.plot(t_pred, mean1, style="pred", label="GPCM")
plt.fill_between(
    t_pred,
    mean1 - 1.96 * B.sqrt(var1),
    mean1 + 1.96 * B.sqrt(var1),
    style="pred",
)
plt.plot(t_pred, mean1 - 1.96 * B.sqrt(var1), style="pred", lw=0.5)
plt.plot(t_pred, mean1 + 1.96 * B.sqrt(var1), style="pred", lw=0.5)

plt.plot(t_pred, mean2, style="pred2", label="GPRVM")
plt.fill_between(
    t_pred,
    mean2 - 1.96 * B.sqrt(var2),
    mean2 + 1.96 * B.sqrt(var2),
    style="pred2",
)
plt.plot(t_pred, mean2 - 1.96 * B.sqrt(var2), style="pred2", lw=0.5)
plt.plot(t_pred, mean2 + 1.96 * B.sqrt(var2), style="pred2", lw=0.5)

plt.xlim(150, 300)
plt.xlabel("Time (Days Into 2012)")
plt.ylabel("Crude Oil (USD)")
tweak()
plt.savefig(wd.file("crude_oil.pdf"))
plt.show()
