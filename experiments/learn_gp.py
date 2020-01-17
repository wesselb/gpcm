import warnings

import lab.tensorflow as B
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wbml.out
import wbml.plot
# noinspection PyUnresolvedReferences
from gpcm.gpcm import GPCM, CGPCM
# noinspection PyUnresolvedReferences
from gpcm.gprv import GPRV
from gpcm.model import train
from gpcm.util import autocorr, estimate_psd
from matplotlib.ticker import FormatStrFormatter
from matrix.util import ToDenseWarning
# noinspection PyUnresolvedReferences
from stheno.tensorflow import GP, EQ, Delta, Matern32, Matern12
from varz import Vars, parametrised, Positive
from varz.tensorflow import minimise_l_bfgs_b

warnings.simplefilter(category=ToDenseWarning, action='ignore')
B.epsilon = 1e-6

# Setup experiment.
n = 250
t = B.linspace(0, 40, n)
noise = 0.01

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
vs_gp = Vars(tf.float64)
lml_gp_opt = -minimise_l_bfgs_b(objective_gp, vs_gp, iters=1000, trace=True)


def construct_model_prototype(vs, Model):
    model = Model(vs=vs, noise=noise, window=window, scale=scale, t=t, n_u=40)
    model.construct(t, y)
    return model


# Build constructors for all models.
models = [
    ('GPCM',
     Vars(tf.float64),
     lambda vs_: construct_model_prototype(vs_, GPCM)),
    ('CGPCM',
     Vars(tf.float64),
     lambda vs_: construct_model_prototype(vs_, CGPCM)),
    ('GP-RV',
     Vars(tf.float64),
     lambda vs_: construct_model_prototype(vs_, GPRV))
]

# Train all models.
for name, vs, construct_model in models:
    with wbml.out.Section(f'Training {name}'):
        construct_model(vs)
        with wbml.out.Section('Variables before optimisation'):
            vs.print()
        elbo = train(construct_model, vs,
                     iters_var=50,
                     iters_var_power=100,
                     iters_no_noise=100,
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
    model = construct_model(vs)
    mu, std = model.predict()

    plt.subplot(3, 1, 1 + i)
    plt.title(name)
    plt.scatter(t, y, c='black', label='Data')
    if hasattr(model, 't_z'):
        plt.scatter(model.t_z, model.t_z*0, s=5, marker='o', c='black')
    plt.plot(t, mu, c='tab:green', label='Prediction')
    plt.fill_between(t, mu - std, mu + std,
                     facecolor='tab:green', alpha=0.15)
    plt.fill_between(t, mu - 2*std, mu + 2*std,
                     facecolor='tab:green', alpha=0.15)
    plt.fill_between(t, mu - 3*std, mu + 3*std,
                     facecolor='tab:green', alpha=0.15)
    plt.plot(t, mu + 3*std + vs['noise']**.5, c='tab:green', ls='--')
    plt.plot(t, mu - 3*std - vs['noise']**.5, c='tab:green', ls='--')
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%.1f$'))
    plt.xlim(min(t), max(t))
    wbml.plot.tweak(legend=True)

plt.tight_layout()
plt.savefig('experiments/learn_gp_functions.pdf')

# Plot kernels.
plt.figure(figsize=(12, 8))

for i, (name, vs, construct_model) in enumerate(models):
    model = construct_model(vs)
    pred = model.predict_kernel()

    # Compute true kernel.
    k_true = kernel(pred.x).mat[0, :]

    # Estimate autocorrelation.
    dt = model.t_u[1] - model.t_u[0]
    lags = int(np.ceil(max(pred.x/dt)))
    t_ac = np.arange(lags + 1)*dt
    k_ac = autocorr(y, lags)

    plt.subplot(3, 1, 1 + i)
    plt.title(name)
    plt.plot(pred.x, pred.mean, c='tab:green', label='Prediction')
    plt.plot(pred.x, k_true, c='black', label='True')
    plt.fill_between(pred.x, pred.err_68_lower, pred.err_68_upper,
                     facecolor='tab:green', alpha=0.15)
    plt.fill_between(pred.x, pred.err_95_lower, pred.err_95_upper,
                     facecolor='tab:green', alpha=0.15)
    plt.fill_between(pred.x, pred.err_99_lower, pred.err_99_upper,
                     facecolor='tab:green', alpha=0.15)
    plt.plot(t_ac, k_ac, c='tab:blue', label='Autocorrelation')
    plt.plot(pred.x, pred.samples, c='tab:red', lw=1)
    plt.scatter(model.t_u, 0*model.t_u, s=5, c='black')
    plt.xlim(0, max(pred.x))
    wbml.plot.tweak(legend=True)

plt.tight_layout()
plt.savefig('experiments/learn_gp_kernels.pdf')

# Plot PSDs.
plt.figure(figsize=(12, 8))

for i, (name, vs, construct_model) in enumerate(models):
    model = construct_model(vs)
    pred = model.predict_psd()

    # Compute true PSD.
    freqs_true, psd_true = estimate_psd(pred.x, kernel(pred.x).mat[0, :])

    # Estimate PSD.
    dt = model.t_u[1] - model.t_u[0]
    lags = int(np.ceil(max(pred.x/dt)))
    t_ac = np.arange(lags + 1)*dt
    k_ac = autocorr(y, lags)
    freqs_ac, psd_ac = estimate_psd(t_ac, k_ac)

    plt.subplot(3, 1, 1 + i)
    plt.title(name)
    plt.plot(pred.x, pred.mean, c='tab:green', label='Prediction')
    plt.plot(freqs_true, psd_true, c='black', label='True')
    plt.fill_between(pred.x, pred.err_68_lower, pred.err_68_upper,
                     facecolor='tab:green', alpha=0.15)
    plt.fill_between(pred.x, pred.err_95_lower, pred.err_95_upper,
                     facecolor='tab:green', alpha=0.15)
    plt.fill_between(pred.x, pred.err_99_lower, pred.err_99_upper,
                     facecolor='tab:green', alpha=0.15)
    plt.plot(freqs_ac, psd_ac, c='tab:blue', label='Autocorrelation')
    plt.plot(pred.x, pred.samples, c='tab:red', lw=1)
    plt.xlim(0, 2)
    plt.ylim(-40, 20)
    wbml.plot.tweak(legend=True)

plt.tight_layout()
plt.savefig('experiments/learn_gp_psds.pdf')

# Plot Fourier features for GP-RV.
plt.figure(figsize=(12, 4))
name, vs, construct_model = models[2]
model = construct_model(vs)
mean, lower, upper = model.predict_fourier()

plt.subplot(1, 2, 1)
plt.title('Cosine Features')
freqs = model.ms/B.to_numpy(model.b - model.a)
inds = np.concatenate(np.where(model.ms == 0) +
                      np.where(model.ms <= model.m_max))
plt.errorbar(freqs[inds], mean[inds], (mean[inds] - lower[inds],
                                       upper[inds] - mean[inds]),
             ls='none', marker='o', capsize=3)
plt.xlabel('Frequency (Hz)')
wbml.plot.tweak()

plt.subplot(1, 2, 2)
plt.title('Sine Features')
freqs = np.maximum(model.ms - model.m_max, 0)/B.to_numpy(model.b - model.a)
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
