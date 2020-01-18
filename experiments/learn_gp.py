import warnings

import lab.tensorflow as B
import matplotlib.pyplot as plt
import numpy as np
import torch
import wbml.plot
from gpcm.gpcm import GPCM, CGPCM
from gpcm.gprv import GPRV
from gpcm.model import train
from gpcm.util import autocorr, estimate_psd
from matplotlib.ticker import FormatStrFormatter
from matrix.util import ToDenseWarning
from stheno.torch import GP, Delta, Matern32
from varz import Vars, parametrised, Positive
from varz.torch import minimise_l_bfgs_b

warnings.simplefilter(category=ToDenseWarning, action='ignore')
B.epsilon = 1e-6

# Setup experiment.
n = 400
t = B.linspace(torch.float64, 0, 40, n)
noise = 0.05

# Setup true model and model.
kernel = Matern32().stretch(1.5)*(lambda x: B.cos(2*B.pi*x/2))
window = 2
scale = window/4

# Sample data.
gp = GP(kernel + noise*Delta())
y = B.flatten(gp(t).sample())


@parametrised
def objective_gp(vs,
                 variance: Positive = 1,
                 scale: Positive = 1.5,
                 noise: Positive = 0.01):
    f = GP(variance*Matern32().stretch(scale)*(lambda x: B.cos(2*B.pi*x/2)))
    e = GP(noise*Delta())
    return -(f + e)(t).logpdf(y)


# Fit a GP.
vs_gp = Vars(torch.float64)
lml_gp_opt = -minimise_l_bfgs_b(objective_gp, vs_gp, iters=1000, trace=True)


def construct_model_prototype(vs, Model):
    model = Model(vs=vs, noise=noise, window=window, scale=scale, t=t, n_u=40)
    model.construct(t, y)
    return model


# Build constructors for all models.
models = [
    ('GPCM',
     Vars(torch.float64),
     lambda vs_: construct_model_prototype(vs_, GPCM)),
    ('CGPCM',
     Vars(torch.float64),
     lambda vs_: construct_model_prototype(vs_, CGPCM)),
    ('GP-RV',
     Vars(torch.float64),
     lambda vs_: construct_model_prototype(vs_, GPRV))
]

# Train all models. Train CGPCM first, because it is the slowest.
for name, vs, construct_model in [models[1], models[0], models[2]]:
    with wbml.out.Section(f'Training {name}'):
        construct_model(vs)
        with wbml.out.Section('Variables before optimisation'):
            vs.print()
        elbo = train(construct_model, vs,
                     iters_var=50,
                     iters_var_power=50,
                     iters_no_noise=50,
                     iters_all=100)
        with wbml.out.Section('Variables after optimisation'):
            vs.print()

        # Print ELBO versus LML of true GP.
        model = construct_model(vs)
        wbml.out.kv('ELBO', model.elbo())
        wbml.out.kv('LML of true GP', gp(t).logpdf(y))
        wbml.out.kv('LML of optimised GP', lml_gp_opt)

# Plot predictions.
plt.figure(figsize=(12, 8))
for i, (name, vs, construct_model) in enumerate(models):
    # Construct model and make predictions.
    model = construct_model(vs)
    mu, std = model.predict()

    plt.subplot(3, 1, 1 + i)
    plt.title(name)

    # Plot data.
    plt.scatter(t, y, c='black', label='Data')

    # Plot inducing models, if the model has them.
    if hasattr(model, 't_z'):
        plt.scatter(model.t_z, model.t_z*0, s=5, marker='o', c='black')

    # Plot the predictions.
    plt.plot(t, mu, c='tab:green', label='Prediction')
    plt.fill_between(t, mu - std, mu + std,
                     facecolor='tab:green', alpha=0.15)
    plt.fill_between(t, mu - 2*std, mu + 2*std,
                     facecolor='tab:green', alpha=0.15)
    plt.fill_between(t, mu - 3*std, mu + 3*std,
                     facecolor='tab:green', alpha=0.15)
    plt.plot(t, mu + 3*std + vs['noise'], c='tab:green', ls='--')
    plt.plot(t, mu - 3*std - vs['noise'], c='tab:green', ls='--')

    # Set limit and format.
    plt.xlim(min(t), max(t))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%.1f$'))
    wbml.plot.tweak(legend=True)

plt.tight_layout()
plt.savefig('experiments/learn_gp_functions.pdf')

# Plot kernels.
plt.figure(figsize=(12, 8))

for i, (name, vs, construct_model) in enumerate(models):
    # Construct model and predict the kernel.
    model = construct_model(vs)
    pred = model.predict_kernel()

    # Compute true kernel.
    k_true = kernel(pred.x).mat[0, :]

    # Estimate autocorrelation.
    t_ac = t - t[0]
    k_ac = autocorr(y)

    plt.subplot(3, 1, 1 + i)
    plt.title(name)

    # Plot inducing points, if the model has them.
    plt.scatter(model.t_u, 0*model.t_u, s=5, c='black')

    # Plot predictions.
    plt.plot(pred.x, pred.mean, c='tab:green', label='Prediction')
    plt.fill_between(pred.x, pred.err_68_lower, pred.err_68_upper,
                     facecolor='tab:green', alpha=0.15)
    plt.fill_between(pred.x, pred.err_95_lower, pred.err_95_upper,
                     facecolor='tab:green', alpha=0.15)
    plt.fill_between(pred.x, pred.err_99_lower, pred.err_99_upper,
                     facecolor='tab:green', alpha=0.15)
    plt.plot(pred.x, pred.samples, c='tab:red', lw=1)

    # Plot the true kernel.
    plt.plot(pred.x, k_true, c='black', label='True')

    # Plot the autocorrelation of the data.
    plt.plot(t_ac, k_ac, c='tab:blue', label='Autocorrelation')

    # Set limits and format.
    plt.xlim(0, max(pred.x))
    wbml.plot.tweak(legend=True)

plt.tight_layout()
plt.savefig('experiments/learn_gp_kernels.pdf')

# Plot PSDs.
plt.figure(figsize=(12, 8))

for i, (name, vs, construct_model) in enumerate(models):
    # Construct compute and predict PSD.
    model = construct_model(vs)
    pred = model.predict_psd()

    # Compute true PSD.
    # TODO: Is `pred.x` okay, or should it be longer?
    freqs_true, psd_true = estimate_psd(pred.x, kernel(pred.x).mat[0, :])

    # Estimate PSD.
    t_ac = t - t[0]
    k_ac = autocorr(y)
    freqs_ac, psd_ac = estimate_psd(t_ac, k_ac)

    plt.subplot(3, 1, 1 + i)
    plt.title(name)

    # Plot predictions.
    plt.plot(pred.x, pred.mean, c='tab:green', label='Prediction')
    plt.fill_between(pred.x, pred.err_68_lower, pred.err_68_upper,
                     facecolor='tab:green', alpha=0.15)
    plt.fill_between(pred.x, pred.err_95_lower, pred.err_95_upper,
                     facecolor='tab:green', alpha=0.15)
    plt.fill_between(pred.x, pred.err_99_lower, pred.err_99_upper,
                     facecolor='tab:green', alpha=0.15)
    plt.plot(pred.x, pred.samples, c='tab:red', lw=1)

    # Plot true PSD.
    plt.plot(freqs_true, psd_true, c='black', label='True')

    # Plot PSD derived from the autocorrelation.
    plt.plot(freqs_ac, psd_ac, c='tab:blue', label='Autocorrelation')

    # Set limits and format.
    plt.xlim(0, 2)
    plt.ylim(-40, 20)
    wbml.plot.tweak(legend=True)

plt.tight_layout()
plt.savefig('experiments/learn_gp_psds.pdf')

# Plot Fourier features for GP-RV.
plt.figure(figsize=(12, 4))

# Construct model and predict the Fourier features.
name, vs, construct_model = models[2]
model = construct_model(vs)
mean, lower, upper = model.predict_fourier()

plt.subplot(1, 2, 1)
plt.title('Cosine Features')
freqs = model.ms/(model.b - model.a)
inds = np.concatenate(np.where(model.ms == 0) +
                      np.where(model.ms <= model.m_max))
plt.errorbar(freqs[inds], mean[inds], (mean[inds] - lower[inds],
                                       upper[inds] - mean[inds]),
             ls='none', marker='o', capsize=3)
plt.xlabel('Frequency (Hz)')
wbml.plot.tweak()

plt.subplot(1, 2, 2)
plt.title('Sine Features')
freqs = np.maximum(model.ms - model.m_max, 0)/(model.b - model.a)
inds = np.concatenate(np.where(model.ms == 0) +
                      np.where(model.ms > model.m_max))
plt.errorbar(freqs[inds], mean[inds], (mean[inds] - lower[inds],
                                       upper[inds] - mean[inds]),
             ls='none', marker='o', capsize=3)
plt.xlabel('Frequency (Hz)')
wbml.plot.tweak()

plt.tight_layout()
plt.savefig('experiments/learn_gp_fourier_features.pdf')

# Show all plots.
plt.show()
