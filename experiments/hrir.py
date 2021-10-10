import argparse

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.metric as metric
import wbml.out as out
from gpcm import GPCM
from scipy.signal import hilbert
from slugify import slugify
from wbml.data.kemar import load
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak


def min_phase(h):
    """Minimum phase transform using the Hilbert transform.

    Args:
        h (vector): Filter to transform.

    Returns:
        vector: Minimum phase filter version of `h`.
    """
    spec = np.fft.fft(h)
    phase = np.imag(-hilbert(np.log(np.abs(spec))))
    return np.fft.ifft(np.abs(spec) * np.exp(1j * phase))


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
    for Model in [GPCM]  # , GPRVM]
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
preds_f_test = []
preds_k = []
if args.predict:
    for model in models:
        # Perform predictions.
        posterior = model.condition(t_train, y_train)
        pred_f = (t_pred,) + normaliser.untransform(posterior.predict(t_pred))
        pred_f_test = (t_test,) + normaliser.untransform(posterior.predict(t_test))
        pred_k = posterior.predict_kernel()
        # Carefully untransform kernel prediction.
        pred_k = (
            pred_k.x,
            pred_k.mean * normaliser._scale,
            pred_k.var * normaliser._scale ** 2,
        )
        # Save predictions.
        preds_f.append(pred_f)
        preds_f_test.append(pred_f_test)
        preds_k.append(pred_k)
        wd.save(pred_f, *model_path(model), "pred_f.pickle")
        wd.save(pred_f_test, *model_path(model), "pred_f_test.pickle")
        wd.save(pred_k, *model_path(model), "pred_k.pickle")
else:
    for model in models:
        preds_f.append(wd.load(*model_path(model), "pred_f.pickle"))
        preds_f_test.append(wd.load(*model_path(model), "pred_f_test.pickle"))
        preds_k.append(wd.load(*model_path(model), "pred_k.pickle"))


def get_kernel_pred(model, scheme):
    i = [(m.name.lower(), m.scheme) for m in models].index((model, scheme))
    t, mean, var = preds_k[i]
    return t, mean, var


def get_pred(model, scheme, test=False):
    i = [(m.name.lower(), m.scheme) for m in models].index((model, scheme))
    model = models[i]
    if test:
        t, mean, var = preds_f_test[i]
    else:
        t, mean, var = preds_f[i]
    # Add observation noise to the prediction.
    var += model().noise * normaliser._scale ** 2
    return t, mean, var


for model in ["gpcm", "gprvm"]:
    print("SMSE", metric.smse(get_pred(model, "structured", test=True)[1], y_test))
    print("SMLL", metric.smll(*get_pred(model, "structured", test=True)[1:], y_test))


mean1, var1 = get_pred("gpcm", "structured")[1:]
mean2, var2 = get_pred("gprvm", "structured")[1:]


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
