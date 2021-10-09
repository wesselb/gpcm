import lab as B
import wbml.out as out
from wbml.plot import tweak
from stheno import EQ, GP
from wbml.experiment import WorkingDirectory
import matplotlib.pyplot as plt

from gpcm import GPCM

out.report_time = True

wd = WorkingDirectory("_experiments", "kernel_cool")

B.epsilon = 1e-8

# Setup experiment.
noise = 0.1
t = B.linspace(0, 40, 200)
t_k = B.linspace(0, 4, 200)

# Setup GPCM models.
window = 2
scale = 0.25
n_u = 80
n_z = 80

# Sample data.
kernel = (lambda x: B.sin(B.pi * x)) * EQ() + (lambda x: B.cos(B.pi * x)) * EQ()
y = B.flatten(GP(kernel)(t, noise).sample())
k = B.flatten(kernel(t_k, 0))

model = GPCM(
    scheme="mean-field",
    window=window,
    scale=scale,
    noise=noise,
    n_u=n_u,
    n_z=n_z,
    t=t,
)
model.fit(t, y, iters=30_000)
k_pred_mf = model.condition(t, y).predict_kernel(t_k)

model = GPCM(
    scheme="structured",
    window=window,
    scale=scale,
    noise=noise,
    n_u=n_u,
    n_z=n_z,
    t=t,
)
model.fit(t, y, iters=30_000)
k_pred_struc = model.condition(t, y).predict_kernel(t_k)

plt.figure(figsize=(5, 3.5))
plt.plot(t_k, k, label="Truth", style="train")

plt.plot(t_k, k_pred_struc.mean, label="Structured", style="pred")
plt.fill_between(
    t_k,
    k_pred_struc.err_95_lower,
    k_pred_struc.err_95_upper,
    style="pred",
)
plt.plot(t_k, k_pred_struc.err_95_upper, style="pred", lw=1)
plt.plot(t_k, k_pred_struc.err_95_lower, style="pred", lw=1)

plt.plot(t_k, k_pred_mf.mean, label="Mean-field", style="pred2")
plt.fill_between(
    t_k,
    k_pred_mf.err_95_lower,
    k_pred_mf.err_95_upper,
    style="pred2",
)
plt.plot(t_k, k_pred_mf.err_95_upper, style="pred2", lw=1)
plt.plot(t_k, k_pred_mf.err_95_lower, style="pred2", lw=1)

plt.xlim(0, 4)
plt.ylim(-0.75, 1.25)
plt.yticks([-0.5, 0, 0.5, 1])

tweak()
plt.savefig(wd.file("smk.pdf"))
plt.show()
