import datetime

import lab as B
import numpy as np
import wbml.out as out
from gpcm import GPCM, CGPCM, RGPCM
from wbml.data.vix import load
from wbml.experiment import WorkingDirectory

# Setup script.
out.report_time = True
B.epsilon = 1e-8
wd = WorkingDirectory("_experiments", f"vix_forecast")

# Setup experiment.
data = load()
# The random year that was selected to train on in the paper is 28 Aug 1991 to 27 Aug
# 1992. We hardcode this to make sure that the random selection isn't changed when the
# seed is changed.
lower = datetime.datetime(1991, 8, 28)
upper = datetime.datetime(2010, 1, 1)
data = data[(data.index >= lower) & (data.index < upper)]
# Convert to days since start. The data type is a timestamp in ns.
t = np.array(data.index - data.index[0], dtype=float) / 1e9 / 3600 / 24
y = np.log(np.array(data.open)).flatten()

# Get training data.
n = 365
n_times = 100
n_forecast = 7
t_train = t[:n]
y_train = y[:n]

# Skip forward `n` days. Get the test data sets.
i = n
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

# Save the data sets.
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
    RGPCM(
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
    # Fit model.
    model.fit(t_train, y_train, iters=10_000)

    # Make predictions for all held-out test sets.
    preds = []
    for t_test, y_test in zip(t_tests, y_tests):
        posterior = model.condition(
            t_test[:-n_forecast],
            (y_test[:-n_forecast] - train_mean) / train_scale,
        )
        mean, var = posterior.predict(t_test[-n_forecast:])
        mean = mean * train_scale + train_mean
        var = (var + model.noise) * train_scale**2
        preds.append((y_test[-n_forecast:], mean, var))
    wd.save(preds, model.name.lower(), "preds.pickle")
