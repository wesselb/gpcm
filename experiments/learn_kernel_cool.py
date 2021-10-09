import lab as B
import wbml.out as out
from wbml.plot import tweak
from stheno import EQ, GP
from wbml.experiment import WorkingDirectory
import matplotlib.pyplot as plt

from gpcm import GPRVM

out.report_time = True

wd = WorkingDirectory("_experiments", "kernel_cool")

B.epsilon = 1e-8

# Setup experiment.
noise = 0.1
t = B.linspace(0, 20, 400)
t_k = B.linspace(0, 4, 200)

# Setup GPCM models.
window = 2
scale = 0.25
n_u = 40
n_z = 80

# Sample data.
kernel = (lambda x: B.sin(2 * B.pi * x)) * EQ() + (lambda x: B.cos(2 * B.pi * x)) * EQ()
y = B.flatten(GP(kernel)(t, noise).sample())
k = B.flatten(kernel(t_k, 0))

model = GPRVM(
    scheme="mean-field",
    window=window,
    scale=scale,
    noise=noise,
    n_u=n_u,
    m_max=n_z // 2,
    t=t,
)
model.fit(t, y)
k_pred_mf = model.condition(t, y).predict_kernel(t_k)

model = GPRVM(
    scheme="structured",
    window=window,
    scale=scale,
    noise=noise,
    n_u=n_u,
    m_max=n_z // 2,
    t=t,
)
model.fit(t, y)
k_pred_struc = model.condition(t, y).predict_kernel(t_k)

plt.figure()
plt.plot(t_k, k, label="Truth", style="train")
plt.plot(t_k, k_pred_mf.mean, label="Mean-field", style="pred")
plt.plot(t_k, k_pred_struc.mean, label="Structured", style="pred2")
tweak()
plt.savefig(wd.file("smk.pdf"))
plt.show()
