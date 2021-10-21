import datetime

import lab as B
import numpy as np
import pandas as pd
import wbml.out as out
from wbml.experiment import WorkingDirectory

from gpcm import GPCM, CGPCM, GPRVM

out.report_time = True
B.epsilon = 1e-8
wd = WorkingDirectory("_experiments", f"forecasting", seed=24)

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
n = 365
n_times = 100
n_forecast = 7
i = np.random.randint(len(t) - n - n_forecast * (n_times + 4) + 1)

# Get training data.
t_train = t[i : i + n]
y_train = y[i : i + n]
t_train = t_train - t_train[0]

i += n
t_tests = []
y_tests = []

for _ in range(n_times):
    t_test = t[i : i + 5 * n_forecast]
    y_test = y[i : i + 5 * n_forecast]
    t_test = t_test - t_test[0]
    t_tests.append(t_test)
    y_tests.append(y_test)
    i += n_forecast


# Setup GPCM models.
window = 7 * 6
scale = 5
n_u = 60
n_z = 150
noise = 0.05

wd.save(
    {
        "t_train": t_train,
        "y_train": y_train,
        "t_tests": t_tests,
        "y_tests": y_tests,
    },
    "data.pickle",
)

# Normalise.
train_mean = y_train.mean()
train_scale = y_train.std()
y_train = (y_train - train_mean) / train_scale

for model in [
    GPRVM(
        window=window,
        scale=scale,
        noise=noise,
        n_u=n_u,
        m_max=n_z // 2,
        t=t_train,
    ),
    GPCM(
        window=window,
        scale=scale,
        noise=noise,
        n_u=n_u,
        n_z=n_z,
        t=t_train,
    ),
    CGPCM(
        window=window,
        scale=scale,
        noise=noise,
        n_u=n_u,
        n_z=n_z,
        t=t_train,
    ),
]:
    model.fit(t_train, y_train, iters=10000)

    preds = []
    for t_test, y_test in zip(t_tests, y_tests):
        posterior = model.condition(
            t_test[:-n_forecast],
            (y_test[:-n_forecast] - train_mean) / train_scale,
        )
        mean, var = posterior.predict(t_test[-n_forecast:])
        mean = mean * train_scale + train_mean
        var = var * train_scale ** 2
        preds.append((y_test[-n_forecast:], mean, var))
    wd.save(preds, model.name.lower(), "preds.pickle")
