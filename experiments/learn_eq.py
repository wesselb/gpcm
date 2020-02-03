import lab.torch as B
import torch
from stheno.torch import GP, Delta, EQ

from gpcm.experiment import setup, run

args, wd = setup('eq')

# Setup experiment.
n = 300
noise = 0.2
t = B.linspace(torch.float64, 0, 20, n)

# Setup true model and GPCM models.
kernel = EQ().stretch(0.5)
window = 1.5
scale = 0.5
n_u = 40
n_z = 40

# Sample data.
gp = GP(kernel + noise*Delta())
y = B.flatten(gp(t).sample())


def comparative_kernel(vs_):
    return vs_.pos(1)*EQ().stretch(vs_.pos(0.5)) + vs_.pos(noise)*Delta()


run(args=args,
    wd=wd,
    noise=noise,
    window=window,
    scale=scale,
    t=t,
    y=y,
    n_u=n_u,
    n_z=n_z,
    true_kernel=kernel,
    true_noisy_kernel=kernel + noise*Delta(),
    comparative_kernel=comparative_kernel)
