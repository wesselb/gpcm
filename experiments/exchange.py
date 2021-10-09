import lab as B
import numpy as np
from probmods.bijection import Normaliser
import wbml.out as out
from wbml.plot import tweak
from stheno import EQ, GP
from wbml.data.exchange import load
from wbml.experiment import WorkingDirectory
import matplotlib.pyplot as plt

from gpcm import GPCM, GPRVM

out.report_time = True
B.epsilon = 1e-8
wd = WorkingDirectory("_experiments", "exchange")

# Setup experiment.
data = load()[0]
t = np.array(data.index)
t = (t - t[0]) * 365  # Day at day zero.
t_pred = B.linspace(min(t), max(t), 500)
y = np.array(data["USD/GBP"])

# Normalise data.
normaliser = Normaliser()
y = normaliser.transform(y)

test_inds = ((50 <= t) & (t <= 100)) | ((200 <= t) & (t <= 250))
t_train = t[~test_inds]
y_train = y[~test_inds]
t_test = t[test_inds]
y_test = y[test_inds]


# Setup GPCM models.
window = 30
scale = 5
n_u = 100
n_z = len(t) // 2

# Setup and fit model.
model = GPRVM(
    scheme="mean-field-ca",
    window=window,
    scale=scale,
    noise=0.05,
    n_u=n_u,
    m_max=n_z // 2,
    t=t,
)
model.fit(t_train, y_train, iters=1000)
mean, var = model.condition(t_train, y_train).predict(t_pred)


plt.figure(figsize=(12, 3))
plt.title(model.name)
plt.scatter(t_train, y_train, style="train")
plt.scatter(t_test, y_test, style="test")
plt.plot(t_pred, mean, style="pred")
plt.fill_between(
    t_pred,
    mean - 1.96 * B.sqrt(var),
    mean + 1.96 * B.sqrt(var),
    style="pred",
)
tweak()
plt.savefig(wd.file("exchange.pdf"))
plt.show()
