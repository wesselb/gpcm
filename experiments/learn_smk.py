import sys

import lab.torch as B
import torch
import wbml.out
from gpcm.experiment import build_models, train_models, plot_compare
from stheno.torch import GP, Delta, EQ
from wbml.experiment import WorkingDirectory

wbml.out.report_time = True

# Setup working directory.
wd = WorkingDirectory('_experiments', 'smk', *sys.argv[1:])

# Setup experiment.
n = 300
noise = 0.1
t = B.linspace(torch.float64, 0, 20, n)

# Setup true model and GPCM models.
kernel = EQ().stretch(1.5)*(lambda x: B.cos(2*B.pi*x*0.5))
window = 3
scale = 0.25

# Sample data.
gp = GP(kernel + noise*Delta())
y = B.flatten(gp(t).sample())


def comparative_kernel(vs_):
    k = vs_.pos(1)*EQ().stretch(vs_.pos(2))
    return k*(lambda x: B.cos(2*B.pi*x*0.5)) + vs_.pos(0.1)*Delta()


# Build and train models.
models = build_models(noise=noise,
                      window=window,
                      scale=scale,
                      t=t,
                      y=y,
                      n_u=50,
                      n_z=50)
train_models(models,
             t=t,
             y=y,
             comparative_kernel=comparative_kernel,
             iters=50)

plot_compare(models,
             t=t,
             y=y,
             wd=wd,
             true_kernel=kernel)
