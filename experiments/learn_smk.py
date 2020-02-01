import argparse

import lab.torch as B
import torch
import wbml.out
from gpcm.experiment import build_models, train_models, analyse_models
from stheno.torch import GP, Delta, EQ
from wbml.experiment import WorkingDirectory

wbml.out.report_time = True

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('path', nargs='*')
parser.add_argument('--quick', action='store_true')
args = parser.parse_args()

# Setup working directory.
wd = WorkingDirectory('_experiments', 'smk', *args.path)

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


models = build_models(noise=noise,
                      window=window,
                      scale=scale,
                      t=t,
                      y=y,
                      n_u=40,
                      n_z=40)

if args.quick:
    samples = train_models(models,
                           burn=200,
                           iters=20,
                           elbo_burn=5,
                           elbo_num_samples=1,
                           num_samples=100)
else:
    samples = train_models(models)

analyse_models(models,
               samples,
               t=t,
               y=y,
               wd=wd,
               true_kernel=kernel,
               true_noisy_kernel=kernel + noise*Delta(),
               comparative_kernel=comparative_kernel)
