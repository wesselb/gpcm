import argparse
from datetime import datetime, timedelta

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.metric as metric
import wbml.out as out
from probmods.bijection import Normaliser
from wbml.data.crude_oil import load
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak, pdfcrop, tex

from gpcm import GPCM, CGPCM, RGPCM

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--predict", action="store_true")
parser.add_argument("--year", type=int, default=2013)
args = parser.parse_args()

# Setup experiment.
out.report_time = True
B.epsilon = 1e-8
tex()
wd = WorkingDirectory("_experiments", "crude_oil", str(args.year))


def first_monday(year):
    """Get the first monday of a year."""
    dt = datetime(year, 1, 1)
    while dt.weekday() != 0:
        dt += timedelta(days=1)
    return dt


# Load and process data.
data = load()
lower = first_monday(args.year)
upper = first_monday(args.year)
data = data[(lower <= data.index) & (data.index < upper)]
t = np.array([(ti - lower).days for ti in data.index], dtype=float)
y = np.array(data.open)
t_pred = B.linspace(min(t), max(t), 500)

# Split data.
test_inds = np.empty(t.shape, dtype=bool)
test_inds.fill(False)
for lower, upper in [
    (
        first_monday(args.year) + i * timedelta(weeks=1),
        first_monday(args.year) + (i + 1) * timedelta(weeks=1),
    )
    for i in range(26, 53)
    if i % 2 == 1
]:
    lower_mask = lower <= data.index
    upper_mask = upper > data.index
    test_inds = test_inds | (lower_mask & upper_mask)
t_train = t[~test_inds]
y_train = y[~test_inds]
t_test = t[test_inds]
y_test = y[test_inds]
# Save data for easier later reference.
wd.save({"train": (t_train, y_train), "test": (t_test, y_test)}, "data.pickle")

# Normalise training data.
normaliser = Normaliser()
y_train = normaliser.transform(y_train)

# Configure GPCM models.
window = 30
scale = 3
n_u = 50
n_z = 150

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
    for Model in [GPCM, CGPCM, RGPCM]
]
if args.train:
    for model in models:
        model.fit(t_train, y_train, iters=20_000)
        model.save(wd.file(model.name.lower(), "model.pickle"))
else:
    for model in models:
        model.load(wd.file(model.name.lower(), "model.pickle"))

# Make and save predictions.
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
            pred_k.var * normaliser._scale**2,
        )
        pred_psd = posterior.predict_psd()
        # Carefully untransform PSD prediction.
        pred_psd = (
            pred_psd.x,
            pred_psd.mean + 20 * B.log(normaliser._scale),
            pred_psd.err_95_lower + 20 * B.log(normaliser._scale),
            pred_psd.err_95_upper + 20 * B.log(normaliser._scale),
        )
        # Save predictions.
        wd.save(pred_f, model.name.lower(), "pred_f.pickle")
        wd.save(pred_f_test, model.name.lower(), "pred_f_test.pickle")
        wd.save(pred_k, model.name.lower(), "pred_k.pickle")
        wd.save(pred_psd, model.name.lower(), "pred_psd.pickle")

# Load predictions.
preds_f = {}
preds_f_test = {}
preds_k = {}
preds_psd = {}
for model in models:
    preds_f[model.name] = wd.load(model.name.lower(), "pred_f.pickle")
    preds_f_test[model.name] = wd.load(model.name.lower(), "pred_f_test.pickle")
    preds_k[model.name] = wd.load(model.name.lower(), "pred_k.pickle")
    preds_psd[model.name] = wd.load(model.name.lower(), "pred_psd.pickle")

# Print performances.
for name in ["GPCM", "CGPCM", "RGPCM"]:
    with out.Section(name):
        t, mean, var = preds_f_test[name]
        out.kv("RMSE", metric.rmse(mean, y_test))
        out.kv("MLL", metric.mll(mean, var, y_test))


def plot_psd(name, y_label=True, style="pred", finish=True):
    freqs, mean, lower, upper = preds_psd[name]
    freqs -= freqs[0]

    inds = freqs <= 0.2
    freqs = freqs[inds]
    mean = mean[inds]
    lower = lower[inds]
    upper = upper[inds]

    if y_label:
        plt.ylabel("PSD (dB)")

    plt.plot(freqs, mean, style=style, label=name)
    plt.fill_between(freqs, lower, upper, style=style)
    plt.plot(freqs, lower, style=style, lw=0.5)
    plt.plot(freqs, upper, style=style, lw=0.5)
    plt.xlim(0, 0.2)
    plt.ylim(0, 60)
    plt.xlabel("Frequency (day${}^{-1}$)")
    if finish:
        tweak()


def plot_compare(name1, name2, y_label=True, y_ticks=True, style2=None):
    _, mean1, var1 = preds_f[name1]
    _, mean2, var2 = preds_f[name2]

    plt.plot(t_pred, mean1, style="pred", label=name1.upper())
    plt.fill_between(
        t_pred,
        mean1 - 1.96 * B.sqrt(var1),
        mean1 + 1.96 * B.sqrt(var1),
        style="pred",
    )
    plt.plot(t_pred, mean1 - 1.96 * B.sqrt(var1), style="pred", lw=0.5)
    plt.plot(t_pred, mean1 + 1.96 * B.sqrt(var1), style="pred", lw=0.5)

    if style2 is None:
        style2 = "pred2"

    plt.plot(t_pred, mean2, style=style2, label=name2.upper())
    plt.fill_between(
        t_pred,
        mean2 - 1.96 * B.sqrt(var2),
        mean2 + 1.96 * B.sqrt(var2),
        style=style2,
    )
    plt.plot(t_pred, mean2 - 1.96 * B.sqrt(var2), style=style2, lw=0.5)
    plt.plot(t_pred, mean2 + 1.96 * B.sqrt(var2), style=style2, lw=0.5)

    plt.scatter(t_train, normaliser.untransform(y_train), style="train", label="Train")
    plt.scatter(t_test, y_test, style="test", label="Test")

    plt.xlim(150, 300)
    plt.xlabel(f"Day of {args.year}")
    if y_label:
        plt.ylabel("Crude Oil (USD)")
    if not y_ticks:
        plt.gca().set_yticklabels([])
    tweak(legend_loc="upper left")


plt.figure(figsize=(12, 5))

plt.subplot2grid((5, 6), (0, 0), colspan=3, rowspan=3)
plot_compare("GPCM", "CGPCM")
plt.subplot2grid((5, 6), (0, 3), colspan=3, rowspan=3)
plot_compare("GPCM", "RGPCM", y_label=False, y_ticks=False, style2="pred3")

plt.subplot2grid((5, 6), (3, 0), colspan=2, rowspan=2)
plot_psd("GPCM")
plt.subplot2grid((5, 6), (3, 2), colspan=2, rowspan=2)
plot_psd("CGPCM", y_label=False, style="pred2")
plt.subplot2grid((5, 6), (3, 4), colspan=2, rowspan=2)
plot_psd("RGPCM", y_label=False, style="pred3")

plt.savefig(wd.file("crude_oil.pdf"))
pdfcrop(wd.file("crude_oil.pdf"))

plt.show()
