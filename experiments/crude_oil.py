import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.out as out
from gpcm import GPCM
from probmods.bijection import Normaliser
from wbml.data.crude_oil import load
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

out.report_time = True
B.epsilon = 1e-8
wd = WorkingDirectory("_experiments", "exchange")


# Setup experiment.
data = load()
data = data[(2012 <= data.index) & (data.index <= 2014)]
t = np.array(data.index)
y = np.array(data.open)

t = (t - t[0]) * 365  # Day at day zero.
t_pred = B.linspace(min(t), max(t), 500)

# Normalise data.
normaliser = Normaliser()
y = normaliser.transform(y)

test_inds = np.empty(t.shape, dtype=bool)
test_inds.fill(False)
for lower, upper in [(100, 140), (220, 260), (420, 500)]:
    test_inds = test_inds | ((lower <= t) & (t <= upper))
t_train = t[~test_inds]
y_train = y[~test_inds]
t_test = t[test_inds]
y_test = y[test_inds]


# Setup GPCM models.
window = 30
scale = 5
n_u = 100
n_z = 200

# Setup and fit model.
model = GPCM(
    scheme="structured",
    window=window,
    scale=scale,
    noise=0.05,
    n_u=n_u,
    n_z=n_z,
    t=t,
)
model.fit(t_train, y_train)
mean, var = model.condition(t_train, y_train).predict(t_pred)


plt.figure(figsize=(12, 3))
# plt.title(model.name)
plt.scatter(t_train, y_train, style="train")
plt.scatter(t_test, y_test, style="test")
# plt.plot(t_pred, mean, style="pred")
# plt.fill_between(
#     t_pred,
#     mean - 1.96 * B.sqrt(var),
#     mean + 1.96 * B.sqrt(var),
#     style="pred",
# )
tweak()
plt.savefig(wd.file("exchange.pdf"))
plt.show()
