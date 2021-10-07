import lab as B
from gpcm.experiment import run, setup
from stheno import Matern12, GP, Delta, Measure

args, wd = setup("kernels")

# Setup experiment.
n = 500 + 1  # Need to add the last point for the call to `linspace`.
noise = 0.05
t = B.linspace(0, 10, n)
t_plot = B.linspace(0, 10, 500)

# Setup true model and GPCM models.
kernel = Matern12()
window = 2
scale = 0.05
n_u = 40
n_z = 100

# Sample data.
m = Measure()
gp_f = GP(kernel, measure=m)
gp_y = gp_f + GP(noise * Delta(), measure=m)
truth, y = map(B.flatten, m.sample(gp_f(t_plot), gp_y(t)))


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
    extend_t_z=True,
    true_kernel=kernel,
    true_noisy_kernel=kernel + noise * Delta(),
    comparative_kernel=comparative_kernel,
    t_plot=t_plot,
    truth=(t_plot, truth),
    x_range={"psd": (0, 3)},
    y_range={"kernel": (-0.5, 1.5), "psd": (-100, 10)},
)
