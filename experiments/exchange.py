import argparse

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.metric as metric
import wbml.out as out
from gpcm import GPRVM
from gpcm.util import min_phase
from slugify import slugify
from wbml.data.exchange import load
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
    wd = WorkingDirectory("server", "_experiments", "exchange", observe=True)
else:
    wd = WorkingDirectory("_experiments", "exchange")

# Load and process data.
data = load()[0]["USD/EUR"]
t = np.array(data.index)
t = (t - t[0]) * 365
y = np.log(np.array(data))

# Normalise.
y = (y - y.mean()) / y.std()

# Configure GPCM models.
window = 30
scale = 3
n_u = 50
n_z = 150


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
    model.fit(t, y, iters=20_000, rate=2e-2, optimise_hypers=1_000)
    model.save(wd.file("model.pickle"))
else:
    model.load(wd.file("model.pickle"))

# Make and save predictions.
if args.predict:
    posterior = model.condition(t, y)
    #  pred_f = (t,) + posterior.predict(t)
    pred_psd = posterior.predict_psd()
    pred_psd = (pred_psd.x, pred_psd.mean,
                pred_psd.err_95_lower, pred_psd.err_95_upper,
                pred_psd.all_samples)
    pred_k = posterior.predict_kernel()
    pred_k = (pred_k.x, pred_k.mean, pred_k.var)
    #  wd.save(pred_f, "pred_f.pickle")
    wd.save(pred_psd, "pred_psd.pickle")
    wd.save(pred_k, "pred_k.pickle")
else:
    pred_f = wd.load("pred_f.pickle")
    pred_psd = wd.load("pred_psd.pickle")
    pred_k = wd.load("pred_k.pickle")


plt.figure()
plt.plot(t_h + 5e-3, h / max(h), label="HRIR", style="train")

t, mean, var = preds_h[0]
err = 1.96 * B.sqrt(var)
plt.plot(t, mean / max(mean), label=models[0].name, style="pred")
plt.fill_between(t, (mean - err) / max(mean), (mean + err) / max(mean), style="pred")
plt.plot(t, (mean - err) / max(mean), style="pred", lw=1)
plt.plot(t, (mean + err) / max(mean), style="pred", lw=1)

# t, mean, var = preds_h[1]
# err = 1.96 * B.sqrt(var)
# plt.plot(t, mean / max(mean), label=models[1].name, style="pred2")
# plt.fill_between(t, (mean - err) / max(mean), (mean + err) / max(mean), style="pred2")
# plt.plot(t, (mean - err) / max(mean), style="pred2", lw=1)
# plt.plot(t, (mean + err) / max(mean), style="pred2", lw=1)

plt.xlim(0, window * 2)
plt.xlabel("Time (ms)")
plt.ylabel("Filter (normalised)")
tweak()
plt.show()


exit()


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

# plt.plot(t_pred, mean2, style="pred2", label="GPRVM")
# plt.fill_between(
#     t_pred,
#     mean2 - 1.96 * B.sqrt(var2),
#     mean2 + 1.96 * B.sqrt(var2),
#     style="pred2",
# )
# plt.plot(t_pred, mean2 - 1.96 * B.sqrt(var2), style="pred2", lw=0.5)
# plt.plot(t_pred, mean2 + 1.96 * B.sqrt(var2), style="pred2", lw=0.5)

plt.xlim(150, 300)
plt.xlabel("Time (Days Into 2012)")
plt.ylabel("Crude Oil (USD)")
tweak()
plt.savefig(wd.file("crude_oil.pdf"))
plt.show()
