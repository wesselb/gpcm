import argparse
from datetime import datetime

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.out as out
from gpcm import GPCM, CGPCM, GPRVM
from probmods.bijection import Normaliser
from slugify import slugify
from wbml.data import date_to_decimal_year
from wbml.data.crude_oil import load
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
    wd = WorkingDirectory("server", "_experiments", "crude_oil", observe=True)
else:
    wd = WorkingDirectory("_experiments", "crude_oil")

# Load and process data.
data = load()
data = data[(2012 <= data.index) & (data.index < 2014)]  # Years 2012 and 2013
t = np.array(data.index)
y = np.array(data.open)
t = (t - t[0]) * 365  # Start at day zero.
t_pred = B.linspace(min(t), max(t), 500)

# Split data.
test_inds = np.empty(t.shape, dtype=bool)
test_inds.fill(False)
for lower, upper in [
    (datetime(2012, 2, 1), datetime(2012, 4, 1)),  # 2012 Feb and March
    (datetime(2012, 10, 1), datetime(2012, 12, 1)),  # 2012 Oct and Nov
    (datetime(2013, 6, 1), datetime(2013, 8, 1)),  # 2013 June and July
]:
    lower_mask = date_to_decimal_year(lower) <= data.index
    upper_mask = date_to_decimal_year(upper) > data.index
    test_inds = test_inds | (lower_mask & upper_mask)
t_train = t[~test_inds]
y_train = y[~test_inds]
t_test = t[test_inds]
y_test = y[test_inds]

# Normalise training data.
normaliser = Normaliser()
y_train = normaliser.transform(y_train)

# Configure GPCM models.
window = 7 * 3
scale = 7
n_u = 50
n_z = 150

# Setup, fit, and save models.
models = [
    Model(
        scheme=scheme,
        window=window,
        scale=scale,
        noise=0.05,
        n_u=n_u,
        n_z=n_z,
        t=t,
    )
    for Model in [GPCM, CGPCM, GPRVM]
    for scheme in ["structured", "mean-field"]
]
if args.train:
    for model in models:
        model.fit(t_train, y_train, iters=50_000)
        model.save(wd.file(slugify(model.name), "model.pickle"))
else:
    for model in models:
        model.load(wd.file(slugify(model.name), "model.pickle"))

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
        pred_k = (
            pred_k.x,
            pred_k.mean * normaliser._scale,
            pred_k.var * normaliser._scale ** 2,
        )
        # Save predictions.
        preds_f.append(pred_f)
        preds_f_test.append(pred_f_test)
        preds_k.append(pred_k)
        wd.save(pred_f, slugify(model.name), "pred_f.pickle")
        wd.save(pred_f_test, slugify(model.name), "pred_f_test.pickle")
        wd.save(preds_k, slugify(model.name), "pred_k.pickle")
else:
    for model in models:
        preds_f.append(wd.load(slugify(model.name), "pred_f.pickle"))
        preds_f_test.append(wd.load(slugify(model.name), "pred_f_test.pickle"))
        preds_k.append(wd.load(slugify(model.name), "pred_k.pickle"))

model = models[0]
mean, var = preds_f[0]

# Plot result.
plt.figure(figsize=(12, 3))
plt.title(model.name)
plt.scatter(t_train, normaliser.untransform(y_train), style="train")
plt.scatter(t_test, y_test, style="test")
plt.plot(t_pred, mean, style="pred")
plt.fill_between(
    t_pred,
    mean - 1.96 * B.sqrt(var),
    mean + 1.96 * B.sqrt(var),
    style="pred",
)
tweak()
plt.savefig(wd.file("crude_oil.pdf"))
plt.show()
