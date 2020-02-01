import warnings

import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import torch
import wbml.plot
from matplotlib.ticker import FormatStrFormatter
from matrix.util import ToDenseWarning
from stheno.torch import GP
from varz import Vars, sequential
from varz.torch import minimise_l_bfgs_b

from .gpcm import GPCM, CGPCM
from .gprv import GPRV
from .model import train
from .util import autocorr, estimate_psd

warnings.simplefilter(category=ToDenseWarning, action='ignore')
B.epsilon = 1e-8

__all__ = ['build_models', 'train_models', 'analyse_models']


def build_models(window,
                 scale,
                 noise,
                 t,
                 y,
                 n_u=40,
                 n_z=None):
    """Construct the GPCM, CGPCM, and GP-RV.

    Args:
        window (scalar): Window length.
        scale (scalar): Length scale of the function.
        t (vector): Time points of data.
        y (vector): Observations.
        n_u (int, optional): Number of inducing points for :math:`h`.
            Defaults to `40`.
        n_z (int, optional): Number of inducing points for :math:`s` or
            equivalent.
    """
    return [('GPCM',
             Vars(torch.float64),
             lambda vs_: GPCM(vs=vs_,
                              noise=noise,
                              window=window,
                              scale=scale,
                              t=t,
                              n_u=n_u,
                              n_z=n_z).construct(t, y)),
            ('CGPCM',
             Vars(torch.float64),
             lambda vs_: CGPCM(vs=vs_,
                               noise=noise,
                               window=window,
                               scale=scale,
                               t=t,
                               n_u=n_u,
                               n_z=n_z).construct(t, y)),
            ('GP-RV',
             Vars(torch.float64),
             lambda vs_: GPRV(vs=vs_,
                              noise=noise,
                              window=window,
                              scale=scale,
                              t=t,
                              n_u=n_u,
                              m_max=int(np.ceil(n_z/2))).construct(t, y))]


def train_models(models,
                 num_samples=1000,
                 **kw_args):
    """Train models.

    Further takes in keyword arguments for :func:`.model.train`.

    Args:
        models (list): Models to train.
        num_samples (int, optional): Number of samples to take.

    Returns:
        list[list[tensor]]: Posterior samples for each of the models.
    """
    samples = []

    for name, vs, construct_model in models:
        with wbml.out.Section(f'Training {name}'):
            construct_model(vs)
            sampler = train(construct_model, vs, **kw_args)

        with wbml.out.Section(f'Sampling {name}'):
            samples.append(sampler.sample(num=num_samples, trace=True))

    return samples


def analyse_models(models,
                   samples,
                   t,
                   y,
                   wd=None,
                   true_kernel=None,
                   true_noisy_kernel=None,
                   comparative_kernel=None):
    """Analyse models.

    Args:
        models (list): Models.
        samples (list[list[tensor]]): Posterior samples for each of the models.
        t (vector): Time points of data.
        y (vector): Observations.
        wd (:class:`wbml.experiment.WorkingDirectory`, optional): Working
            directory to save the plots to.
        true_kernel (:class:`stheno.Kernel`, optional): True kernel that
            generated the data, not including noise.
        true_noisy_kernel (:class:`stheno.Kernel`, optional): True kernel that
            generated the data, including noise.
        comparative_kernel (function, optional): A function that takes in a
            variable container and gives back a kernel. A GP with this
            kernel will be trained on the data to compute a likelihood that
            will be compared to the ELBOs.
    """

    # Print the learned variables.
    with wbml.out.Section('Variables after optimisation'):
        for name, vs, construct_model in models:
            with wbml.out.Section(name):
                vs.print()

    analyse_elbos(models,
                  samples,
                  t=t,
                  y=y,
                  true_noisy_kernel=true_noisy_kernel,
                  comparative_kernel=comparative_kernel)
    analyse_plots(models,
                  samples,
                  t=t,
                  y=y,
                  wd=wd,
                  true_kernel=true_kernel)


def analyse_elbos(models,
                  samples,
                  t,
                  y,
                  true_noisy_kernel=None,
                  comparative_kernel=None):
    """Compare ELBOs of models.

    Args:
        models (list): Models to train.
        samples (list[list[tensor]]): Posterior samples for each of the models.
        t (vector): Time points of data.
        y (vector): Observations.
        true_noisy_kernel (:class:`stheno.Kernel`, optional): True kernel that
            generated the data, including noise.
        comparative_kernel (function, optional): A function that takes in a
            variable container and gives back a kernel. A GP with this
            kernel will be trained on the data to compute a likelihood that
            will be compared to the ELBOs.
    """

    # Print LML under true GP if the true kernel is given.
    if true_noisy_kernel:
        wbml.out.kv('LML under true GP', GP(true_noisy_kernel)(t).logpdf(y))

    # Print LML under a trained GP if a comparative kernel is given.
    if comparative_kernel:
        def objective(vs_):
            gp = GP(sequential(comparative_kernel)(vs_))
            return -gp(t).logpdf(y)

        # Fit the GP.
        vs = Vars(torch.float64)
        lml_gp_opt = -minimise_l_bfgs_b(objective, vs, iters=1000)

        # Print likelihood.
        wbml.out.kv('LML under optimised GP', lml_gp_opt)

    # Estimate ELBOs.
    with wbml.out.Section('ELBOs'):
        for i, (name, vs, construct_model) in enumerate(models):
            model = construct_model(vs)
            wbml.out.kv(name, model.elbo(samples[i]))


def analyse_plots(models,
                  samples,
                  t,
                  y,
                  true_kernel=None,
                  wd=None):
    """Analyse models in plots.

    Args:
        models (list): Models to train.
        samples (list[list[tensor]]): Posterior samples for each of the models.
        t (vector): Time points of data.
        y (vector): Observations.
        true_kernel (:class:`stheno.Kernel`, optional): True kernel that
            generates the data for comparison.
        wd (:class:`wbml.experiment.WorkingDirectory`, optional): Working directory
            to save the plots to.
    """
    # Plot predictions.
    plt.figure(figsize=(12, 8))

    for i, (name, vs, construct_model) in enumerate(models):
        # Construct model and make predictions.
        model = construct_model(vs)
        mu, std = model.predict(samples[i])

        plt.subplot(3, 1, 1 + i)
        plt.title(f'Function ({name})')

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
        error = 2*B.sqrt(vs['noise'] + std**2)  # Model and noise error.
        plt.plot(t, mu + error, c='tab:green', ls='--')
        plt.plot(t, mu - error, c='tab:green', ls='--')

        # Set limit and format.
        plt.xlim(min(t), max(t))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%.1f$'))
        wbml.plot.tweak(legend=True)

    plt.tight_layout()
    if wd:
        plt.savefig(wd.file('prediction_function.pdf'))

    # Plot kernels.
    plt.figure(figsize=(12, 8))

    for i, (name, vs, construct_model) in enumerate(models):
        # Construct model and predict the kernel.
        model = construct_model(vs)
        with wbml.out.Section(f'Predicting kernel for {name}'):
            pred = model.predict_kernel(samples[i])

        # Compute true kernel.
        if true_kernel:
            k_true = true_kernel(pred.x).mat[0, :]

        # Estimate autocorrelation.
        t_ac = t - t[0]
        k_ac = autocorr(y, normalise=False)

        plt.subplot(3, 1, 1 + i)
        plt.title(f'Kernel ({name})')

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
        if true_kernel:
            plt.plot(pred.x, k_true, c='black', label='True')

        # Plot the autocorrelation of the data.
        inds = t_ac <= max(pred.x)  # Only plot visible bit.
        plt.plot(t_ac[inds], k_ac[inds], c='tab:blue', label='Autocorrelation')

        # Set limits and format.
        plt.xlim(0, max(pred.x))
        plt.ylim(-1, 1)
        wbml.plot.tweak(legend=True)

    plt.tight_layout()
    if wd:
        plt.savefig(wd.file('prediction_kernel.pdf'))

    # Plot PSDs.
    plt.figure(figsize=(12, 8))

    for i, (name, vs, construct_model) in enumerate(models):
        # Construct compute and predict PSD.
        model = construct_model(vs)
        with wbml.out.Section(f'Predicting PSD for {name}'):
            pred = model.predict_psd(samples[i])

        # Compute true PSD.
        if true_kernel:
            # TODO: Is `pred.x` okay, or should it be longer?
            freqs_true, psd_true = \
                estimate_psd(pred.x, true_kernel(pred.x).mat[0, :])

        # Estimate PSD.
        t_ac = t - t[0]
        k_ac = autocorr(y, normalise=False)
        freqs_ac, psd_ac = estimate_psd(t_ac, k_ac)

        plt.subplot(3, 1, 1 + i)
        plt.title(f'PSD ({name})')

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
        if true_kernel:
            plt.plot(freqs_true, psd_true, c='black', label='True')

        # Plot PSD derived from the autocorrelation.
        plt.plot(freqs_ac, psd_ac, c='tab:blue', label='Autocorrelation')

        # Set limits and format.
        plt.xlim(0, 2)
        plt.ylim(0, 3)
        wbml.plot.tweak(legend=True)

    plt.tight_layout()
    if wd:
        plt.savefig(wd.file('prediction_psd.pdf'))

    # Plot Fourier features for GP-RV.
    plt.figure(figsize=(12, 8))

    for i, (name, vs, construct_model) in enumerate(models):
        # Construct model.
        model = construct_model(vs)

        # Predict Fourier features if it is a GP-RV.
        if isinstance(model, GPRV):
            mean, lower, upper = model.predict_fourier(samples[i])
        else:
            continue

        plt.subplot(3, 2, 1 + 2*i)
        plt.title(f'Cosine Features ({name})')
        freqs = model.ms/(model.b - model.a)
        inds = np.concatenate(np.where(model.ms == 0) +
                              np.where(model.ms <= model.m_max))
        plt.errorbar(freqs[inds], mean[inds], (mean[inds] - lower[inds],
                                               upper[inds] - mean[inds]),
                     ls='none', marker='o', capsize=3)
        plt.xlabel('Frequency (Hz)')
        wbml.plot.tweak(legend=False)

        plt.subplot(3, 2, 2 + 2*i)
        plt.title(f'Sine Features ({name})')
        freqs = np.maximum(model.ms - model.m_max, 0)/(model.b - model.a)
        inds = np.concatenate(np.where(model.ms == 0) +
                              np.where(model.ms > model.m_max))
        plt.errorbar(freqs[inds], mean[inds], (mean[inds] - lower[inds],
                                               upper[inds] - mean[inds]),
                     ls='none', marker='o', capsize=3)
        plt.xlabel('Frequency (Hz)')
        wbml.plot.tweak(legend=False)

    plt.tight_layout()
    if wd:
        plt.savefig(wd.file('prediction_fourier.pdf'))
