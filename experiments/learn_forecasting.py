import argparse
import datetime

import lab as B
import numpy as np
import pandas as pd
import wbml.out as out
from wbml.experiment import WorkingDirectory

from gpcm import GPCM, CGPCM, GPRVM

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)
args = parser.parse_args()

out.report_time = True
B.epsilon = 1e-8
wd = WorkingDirectory("_experiments", f"forecasting/{args.seed}", seed=args.seed)

# Setup experiment.
data = pd.read_parquet("experiments/vix.parquet")
data = data.set_index("date")
lower = datetime.datetime(1990, 1, 1)
upper = datetime.datetime(2010, 1, 1)
data = data[(data.index >= lower) & (data.index < upper)]
# Convert to days since start. The data type is a timestamp in ns.
t = np.array(data.index - data.index[0], dtype=float) / 1e9 / 3600 / 24
y = np.log(np.array(data.open)).flatten()

# Sample random subset.
n = 180
n_forecast = 7
i = np.random.randint(len(t) - n - n_forecast + 1)
t_train = t[i : i + n]
y_train = y[i : i + n]
t_test = t[i + n : i + n + n_forecast]
y_test = y[i + n : i + n + n_forecast]
t_all = t[i : i + n + n_forecast]

# Normalise.
train_scale = y_train.std()
train_mean = y_train.mean()
y_train = (y_train - train_scale) / train_mean

# Setup GPCM models.
window = 7 * 3
scale = 3
n_u = 30
n_z = 60

wd.save(
    {
        "t_train": t_train,
        "y_train": y_train,
        "t_test": t_train,
        "y_test": y_train,
    },
    "data.pickle",
)

for model in [
    GPCM(
        window=window,
        scale=scale,
        noise=0.05,
        n_u=n_u,
        n_z=n_z,
        t=t,
    ),
    CGPCM(
        window=window,
        scale=scale,
        noise=0.05,
        n_u=n_u,
        n_z=n_z,
        t=t,
    ),
    GPRVM(
        window=window,
        scale=scale,
        noise=0.05,
        n_u=n_u,
        m_max=n_z // 2,
        t=t,
    ),
]:
    # Fit model and predict function and kernel.
    model.fit(t, y, iters=10_000)
    posterior = model.condition(t, y)

    mean, var = posterior.predict(t_all)
    mean = mean * train_scale + train_mean
    var = var * train_scale ** 2
    wd.save((t_all, mean, var), "all_pred.pickle")

    mean, var = posterior.predict(t_test)
    mean = mean * train_scale + train_mean
    var = var * train_scale ** 2
    wd.save((t_test, mean, var), "test_pred.pickle")

    mean, var = posterior.predict(t_train)
    mean = mean * train_scale + train_mean
    var = var * train_scale ** 2
    wd.save((t_train, mean, var), "train_pred.pickle")
