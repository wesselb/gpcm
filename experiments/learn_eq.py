import sys

import lab.torch as B
import torch
import wbml.out
from stheno.torch import GP, Delta, EQ
from wbml.experiment import WorkingDirectory

from gpcm.experiment import build_models, train_models, plot_compare

wbml.out.report_time = True

# Setup working directory.
wd = WorkingDirectory('_experiments', 'eq', *sys.argv[1:])

# Setup experiment.
n = 800
noise = 1.0
t = B.linspace(torch.float64, 0, 40, n)

# Setup true model and GPCM models.
kernel = EQ().stretch(0.5)
window = 1.5
scale = 0.5

# Sample data.
gp = GP(kernel + noise*Delta())
y = B.flatten(gp(t).sample())


def comparative_kernel(vs_):
    return vs_.pos(1)*EQ().stretch(vs_.pos(0.5)) + vs_.pos(1)*Delta()


# Build and train models.
models = build_models(noise=noise,
                      window=window,
                      scale=scale,
                      t=t,
                      y=y,
                      n_u=40,
                      n_z=80)
train_models(models,
             t=t,
             y=y,
             comparative_kernel=comparative_kernel,
             iters_pre=100,
             iters_fixed_noise=200,
             iters=200)

plot_compare(models,
             t=t,
             y=y,
             wd=wd,
             true_kernel=kernel)
