import lab as B
from stheno import EQ, GP, Delta, Measure

from gpcm.experiment import run, setup

args, wd = setup("eq")

# Setup experiment.
n = 880
noise = 0.1
t = B.linspace(-44, 44, n)
t_plot = B.linspace(44, 44, 500)

# Setup true model and GPCM models.
kernel = EQ()
window = 2
scale = 1
n_u = 40
n_z = 88

# Sample data.
m = Measure()
gp_f = GP(kernel, measure=m)
gp_y = gp_f + GP(noise * Delta(), measure=m)
truth, y = map(B.flatten, m.sample(gp_f(t_plot), gp_y(t)))

# Remove region [-8.8, 8.8].
inds = ~((t >= -8.8) & (t <= 8.8))
t = t[inds]
y = y[inds]


def comparative_kernel(vs_):
    return vs_.pos(1) * kernel.stretch(vs_.pos(1.0)) + vs_.pos(noise) * Delta()


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
    x_range={"psd": (0, 0.5)},
    y_range={"kernel": (-0.5, 1.5), "psd": (-20, 10)},
)
