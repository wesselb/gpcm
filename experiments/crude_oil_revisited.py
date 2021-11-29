import argparse
from datetime import datetime, timedelta

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.metric as metric
import wbml.out as out
from matplotlib.patches import Ellipse
from probmods.bijection import Normaliser
from slugify import slugify
from wbml.data import date_to_decimal_year
from wbml.data.crude_oil import load
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak, pdfcrop

from gpcm import GPCM, CGPCM, GPRVM

# Setup experiment.
out.report_time = True
B.epsilon = 1e-8

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--predict", action="store_true")
parser.add_argument("--year", required=True, type=int)
args = parser.parse_args()
year = args.year

wd = WorkingDirectory("_experiments", "crude_oil_rev", str(year))

# Find first Monday of year.
current = datetime(year, 1, 1)
if current.weekday() > 0:
    current += (7 - current.weekday()) * timedelta(days=1)

# Load and split off year.
data = load()
data = data[(date_to_decimal_year(current) <= data.index) & (data.index < year + 1)]
data.dropna(inplace=True)


def convert_t_y(df):
    return (np.array(df.index) - year) * 365, np.array(df.open)


# Split off train set.
week = timedelta(days=7)
current += 26 * week
train = data[data.index < date_to_decimal_year(current)]
t_train, y_train = convert_t_y(train)

# Set off eval sets.
evals = []
while date_to_decimal_year(current) < year + 1:
    this_train = data[data.index < date_to_decimal_year(current)]
    this_test = data[
        (date_to_decimal_year(current) <= data.index)
        & (data.index < date_to_decimal_year(current + week))
    ]
    if this_test.shape[0] > 0:
        evals.append((convert_t_y(this_train), convert_t_y(this_test)))
        current += week
    else:
        break

wd.save("data.pickle", ((t_train, y_train), evals))

# Normalise training data.
normaliser = Normaliser()
y_train = normaliser.transform(y_train)

# Configure GPCM models.
window = 30
scale = 3
n_u = 50
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
        extend_t_z=True,
        t=(0, 365),
    )
    for Model in [CGPCM, GPCM, GPRVM]
]
if args.train:
    for model in models:
        model.fit(t_train, y_train, iters=5_000)
        model.save(wd.file(*model_path(model), "model.pickle"))
else:
    for model in models:
        model.load(wd.file(*model_path(model), "model.pickle"))

# Make and save predictions.
preds_f = []
preds_k = []
preds_psd = []
preds_f_test = []
if args.predict:
    # Predict based on first half.
    for model in models:
        # Perform predictions.
        posterior = model.condition(t_train, y_train)
        pred_f = (t_train,) + normaliser.untransform(posterior.predict(t_train))
        pred_k = posterior.predict_kernel()
        # Carefully untransform kernel prediction.
        pred_k = (
            pred_k.x,
            pred_k.mean * normaliser._scale,
            pred_k.var * normaliser._scale ** 2,
        )
        pred_psd = posterior.predict_psd()
        # Carefully untransform kernel prediction.
        pred_psd = (
            pred_psd.x,
            pred_psd.mean + 20 * B.log(normaliser._scale),
            pred_psd.err_95_lower + 20 * B.log(normaliser._scale),
            pred_psd.err_95_upper + 20 * B.log(normaliser._scale),
        )
        # Save predictions.
        preds_f.append(pred_f)
        preds_k.append(pred_k)
        preds_psd.append(pred_psd)
        wd.save(pred_f, *model_path(model), "pred_f.pickle")
        wd.save(pred_k, *model_path(model), "pred_k.pickle")
        wd.save(pred_psd, *model_path(model), "pred_psd.pickle")

    # Perform predictions for second half.
    for model in models:
        pred_f_test = []
        for (t_train, y_train), (t_test, _) in evals:
            # Perform predictions.
            posterior = model.condition(t_train, y_train)
            pred = (t_test,) + normaliser.untransform(posterior.predict(t_test))
            pred_f_test.append(pred)
        preds_f_test.append(pred_f_test)
        wd.save(pred_f_test, *model_path(model), "pred_f_test.pickle")
else:
    for model in models:
        preds_f.append(wd.load(*model_path(model), "pred_f.pickle"))
        preds_k.append(wd.load(*model_path(model), "pred_k.pickle"))
        preds_psd.append(wd.load(*model_path(model), "pred_psd.pickle"))
        preds_f_test.append(wd.load(*model_path(model), "pred_f_test.pickle"))


def get_kernel_pred(model, scheme):
    i = [(m.name.lower(), m.scheme) for m in models].index((model, scheme))
    t, mean, var = preds_k[i]
    return t, mean, var


def get_psd_pred(model, scheme):
    i = [(m.name.lower(), m.scheme) for m in models].index((model, scheme))
    return preds_psd[i]


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


print("Structured")
for model in ["gpcm", "cgpcm", "gprvm"]:
    print(model)
    print("RMSE", metric.rmse(get_pred(model, "structured", test=True)[1], y_test))
    print("MLL", metric.mll(*get_pred(model, "structured", test=True)[1:], y_test))

print("MF")
for model in ["gpcm", "cgpcm", "gprvm"]:
    print(model)
    print("RMSE", metric.rmse(get_pred(model, "mean-field", test=True)[1], y_test))
    print("MLL", metric.mll(*get_pred(model, "mean-field", test=True)[1:], y_test))


def model_name_map(name):
    if name == "GPRVM":
        return "RGPCM"
    else:
        return name


def plot_psd(model, y_label=True):
    freqs, mean, lower, upper = get_psd_pred(model, "structured")
    freqs -= freqs[0]

    inds = freqs <= 0.2
    freqs = freqs[inds]
    mean = mean[inds]
    lower = lower[inds]
    upper = upper[inds]

    if y_label:
        plt.ylabel("PSD (dB)")

    plt.plot(freqs, mean, style="pred", label=model_name_map(model.upper()))
    plt.fill_between(
        freqs,
        lower,
        upper,
        style="pred",
    )
    plt.plot(freqs, lower, style="pred", lw=0.5)
    plt.plot(freqs, upper, style="pred", lw=0.5)
    plt.xlim(0, 0.2)
    plt.ylim(0, 60)
    plt.xlabel("Frequency (day${}^{-1}$)")
    tweak()


def plot_compare(model1, model2, y_label=True, y_ticks=True):
    plt.scatter(t_train, normaliser.untransform(y_train), style="train", label="Train")
    plt.scatter(t_test, y_test, style="test", label="Test")

    mean1, var1 = get_pred(model1, "structured")[1:]
    mean2, var2 = get_pred(model2, "structured")[1:]

    plt.plot(t_pred, mean1, style="pred", label=model_name_map(model1.upper()))
    plt.fill_between(
        t_pred,
        mean1 - 1.96 * B.sqrt(var1),
        mean1 + 1.96 * B.sqrt(var1),
        style="pred",
    )
    plt.plot(t_pred, mean1 - 1.96 * B.sqrt(var1), style="pred", lw=0.5)
    plt.plot(t_pred, mean1 + 1.96 * B.sqrt(var1), style="pred", lw=0.5)

    plt.plot(t_pred, mean2, style="pred2", label=model_name_map(model2.upper()))
    plt.fill_between(
        t_pred,
        mean2 - 1.96 * B.sqrt(var2),
        mean2 + 1.96 * B.sqrt(var2),
        style="pred2",
    )
    plt.plot(t_pred, mean2 - 1.96 * B.sqrt(var2), style="pred2", lw=0.5)
    plt.plot(t_pred, mean2 + 1.96 * B.sqrt(var2), style="pred2", lw=0.5)

    # Add some circles to zoom into on interesting features. These are manually placed.
    plt.gca().add_artist(
        Ellipse((190, 85), 5 * 2, 3 * 2, ec="black", fc="none", lw=1, zorder=10)
    )
    plt.gca().add_artist(
        Ellipse((219, 93), 5 * 2, 3 * 2, ec="black", fc="none", lw=1, zorder=10)
    )
    plt.gca().add_artist(
        Ellipse((260, 92), 5 * 2, 3 * 2, ec="black", fc="none", lw=1, zorder=10)
    )

    plt.xlim(150, 300)
    plt.xlabel("Day of 2012")
    if y_label:
        plt.ylabel("Crude Oil (USD)")
    if not y_ticks:
        plt.gca().set_yticklabels([])
    tweak(legend_loc="upper left")


# Plot result.
plt.figure(figsize=(12, 5))

plt.subplot2grid((5, 6), (0, 0), colspan=3, rowspan=3)
plot_compare("gpcm", "cgpcm")
plt.subplot2grid((5, 6), (0, 3), colspan=3, rowspan=3)
plot_compare("gpcm", "gprvm", y_label=False, y_ticks=False)

plt.subplot2grid((5, 6), (3, 0), colspan=2, rowspan=2)
plot_psd("gpcm")
plt.subplot2grid((5, 6), (3, 2), colspan=2, rowspan=2)
plot_psd("cgpcm", y_label=False)
plt.subplot2grid((5, 6), (3, 4), colspan=2, rowspan=2)
plot_psd("gprvm", y_label=False)

plt.savefig(wd.file("crude_oil.pdf"))
pdfcrop(wd.file("crude_oil.pdf"))

plt.show()
