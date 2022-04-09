from datetime import datetime, timedelta

import lab as B
import numpy as np
import wbml.out as out
from probmods import Normaliser
from wbml.data.vix import load
from wbml.experiment import WorkingDirectory

from gpcm import GPCM, CGPCM, RGPCM

# Setup script.
out.report_time = True
B.epsilon = 1e-8
wd = WorkingDirectory("_experiments", f"vix_forecast")

# Setup experiment.
data = load()


def get_data(lower, upper):
    """Get data for a certain time range."""
    df = data[(data.index >= lower) & (data.index < upper)]
    #  The data type is a timestamp in ns.
    t = np.array(df.index, dtype=float) / 1e9 / 3600 / 24 / 365
    y = np.log(np.array(df.open)).flatten()
    return t, y


# Train on the year of 2015.
t_train, y_train = get_data(datetime(2015, 1, 1), datetime(2016, 1, 1))
t_train -= t_train[0]  # Count since start.

# Get the test data sets.
tests = []
for i in range(100):
    t_test1, y_test1 = get_data(
        datetime(2016, 1, 1) + i * timedelta(weeks=1),
        datetime(2016, 1, 1) + (i + 4) * timedelta(weeks=1),
    )
    t_test2, y_test2 = get_data(
        datetime(2016, 1, 1) + (i + 4) * timedelta(weeks=1),
        datetime(2016, 1, 1) + (i + 5) * timedelta(weeks=1),
    )
    # Count since beginning of conditioning window.
    t_test2 -= t_test1[0]
    t_test1 -= t_test1[0]
    tests.append(((t_test1, y_test1), (t_test2, y_test2)))
# Save the data sets.
wd.save(
    {"t_train": t_train, "y_train": y_train, "tests": tests},
    "data.pickle",
)

# Setup GPCM models.
window = 7 * 6
scale = 5
n_u = 60
n_z = 150
noise = 0.05


# Normalise.
normaliser = Normaliser()
y_train = normaliser.transform(y_train)

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
    with out.Progress("Making predictions", total=len(tests)) as progress:
        for (t_test1, y_test1), (t_test2, y_test2) in tests:
            posterior = model.condition(t_test1, normaliser.transform(y_test1))
            mean, var = normaliser.untransform(posterior.predict(t_test2))
            preds.append((y_test2, mean, var))
            progress()
    wd.save(preds, model.name.lower(), "preds.pickle")
