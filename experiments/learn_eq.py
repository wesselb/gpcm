import lab.torch as B
import torch
from stheno.torch import Measure, GP, Delta, EQ

from gpcm.experiment import setup, run

args, wd = setup("eq")

# Setup experiment.
n = 880 + 1  # Need to add the last point for the call to `linspace`.
noise = 1.0
t = B.linspace(torch.float64, -44, 44, n)
t_plot = B.linspace(torch.float64, -44, 44, 500)

# Setup true model and GPCM models.
kernel = EQ()
window = 2
scale = 0.7
n_u = 30
n_z = 88

# Sample data.
m = Measure()
gp_f = GP(kernel, measure=m)
gp_y = gp_f + GP(noise * Delta(), measure=m)
truth, y = map(B.flatten, m.sample(gp_f(t_plot), gp_y(t)))

# Remove region [-4.4, 4.4].
inds = ~((t >= -4.4) & (t <= 4.4))
t = t[inds]
y = y[inds]


def comparative_kernel(vs_):
    return vs_.pos(1) * EQ().stretch(vs_.pos(1.0)) + vs_.pos(noise) * Delta()


run(
    args=args,
    wd=wd,
    noise=noise,
    window=window,
    scale=scale,
    t=t,
    y=y,
    n_u=n_u,
    n_z=n_z,
    true_kernel=kernel,
    true_noisy_kernel=kernel + noise * Delta(),
    comparative_kernel=comparative_kernel,
    t_plot=t_plot,
    truth=(t_plot, truth),
    y_range={"kernel": (-0.5, 1.5)},
)
