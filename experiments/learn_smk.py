import lab.torch as B
import torch
from stheno.torch import GP, Delta, EQ

from gpcm.experiment import setup, run

args, wd = setup('smk')

# Setup experiment.
n = 300
noise = 0.05
t = B.linspace(torch.float64, 0, 20, n)

# Setup true model and GPCM models.
kernel = EQ().stretch(1.5)*(lambda x: B.cos(2*B.pi*x*0.5)) + \
         EQ().stretch(1.5)*(lambda x: B.sin(2*B.pi*x*0.5))
window = 2
scale = 0.25

# Sample data.
gp = GP(kernel + noise*Delta())
y = B.flatten(gp(t).sample())


def comparative_kernel(vs_):
    k = vs_.pos(1)*EQ().stretch(vs_.pos(2))
    return k*(lambda x: B.cos(2*B.pi*x*0.5)) + \
           k*(lambda x: B.sin(2*B.pi*x*0.5)) + \
           vs_.pos(0.1)*Delta()


run(args=args,
    wd=wd,
    noise=noise,
    window=window,
    scale=scale,
    t=t,
    y=y,
    n_u=50,
    n_z=50,
    true_kernel=kernel,
    true_noisy_kernel=kernel + noise*Delta(),
    comparative_kernel=comparative_kernel)
