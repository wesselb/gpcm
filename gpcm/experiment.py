import argparse
import warnings

import lab.torch as B
import matplotlib.pyplot as plt
import numpy as np
import torch
import wbml.out
import wbml.plot
from matplotlib.ticker import FormatStrFormatter
from matrix.util import ToDenseWarning
from stheno.torch import GP
from varz import Vars, sequential
from varz.torch import minimise_l_bfgs_b
from wbml.experiment import WorkingDirectory

from .gpcm import GPCM, CGPCM
from .gprv import GPRV
from .model import train_vi, train_laplace
from .util import autocorr, estimate_psd

warnings.simplefilter(category=ToDenseWarning, action="ignore")
B.epsilon = 1e-8
wbml.out.report_time = True

__all__ = ["setup", "run", "build_models", "train_models", "analyse_models"]


def setup(name):
    """Setup an experiment.

    Args:
        name (str): Name of the experiment.

    Returns:
        tuple[:class:`argparse.Namespace`,
              :class:`wbml.experiment.WorkingDirectory`]: Tuple containing
            the parsed arguments and the working directory.
    """
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="*")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--instant", action="store_true")
    parser.add_argument("--fix-noise", action="store_true")
    parser.add_argument(
        "--train-method",
        choices=["vi", "laplace"],
        default="vi",
        nargs="?",
    )
    parser.add_argument(
        "--model",
        choices=["gpcm", "gprv", "cgpcm"],
        default=["gpcm", "gprv", "cgpcm"],
        nargs="+",
    )
    args = parser.parse_args()

    # Setup working directory.
    wd = WorkingDirectory("_experiments", name, *args.path)

    return args, wd


def run(args, wd, noise, window, scale, t, y, n_u, n_z, **kw_args):
    """Run an experiment.

    Further takes in keyword arguments for :func:`.experiment.analyse_models`.

    Args:
        args (:class:`.argparse.Namespace`): Parser arguments.
        wd (:class:`wbml.experiment.WorkingDirectory`): Working directory.
        noise (scalar): Noise variance.
        window (scalar): Window length.
        scale (scalar): Length scale of the function.
        t (vector): Time points of data.
        y (vector): Observations.
        n_u (int): Number of inducing points for :math:`h`.
        n_z (int): Number of inducing points for :math:`s` or equivalent.
    """
    models = build_models(
        args.model, noise=noise, window=window, scale=scale, t=t, y=y, n_u=n_u, n_z=n_z
    )

    # Get the training function.
    train = {"vi": train_vi, "laplace": train_laplace}[args.train_method]

    if args.instant:
        dists = train_models(train, models, wd=wd, iters=0, fix_noise=args.fix_noise)
    else:
        if args.quick:
            dists = train_models(
                train, models, wd=wd, iters=20, fix_noise=args.fix_noise
            )
        else:
            dists = train_models(
                train, models, wd=wd, iters=200, fix_noise=args.fix_noise
            )

    analyse_models(models, dists, t=t, y=y, wd=wd, **kw_args)


def build_models(names, window, scale, noise, t, y, n_u=40, n_z=None):
    """Construct the GPCM, CGPCM, and GP-RV.

    Args:
        names (list[str]): Names of models to build.
        window (scalar): Window length.
        scale (scalar): Length scale of the function.
        t (vector): Time points of data.
        y (vector): Observations.
        n_u (int, optional): Number of inducing points for :math:`h`. Defaults to `40`.
        n_z (int, optional): Number of inducing points for :math:`s` or equivalent.
    """
    models = []

    if "gpcm" in names:
        names = set(names) - {"gpcm"}
        models.append(
            (
                "GPCM",
                Vars(torch.float64),
                lambda vs_: GPCM(
                    vs=vs_,
                    noise=noise,
                    window=window,
                    scale=scale,
                    t=t,
                    n_u=n_u,
                    n_z=n_z,
                ).construct(t, y),
            )
        )

    if "cgpcm" in names:
        names = set(names) - {"cgpcm"}
        models.append(
            (
                "CGPCM",
                Vars(torch.float64),
                lambda vs_: CGPCM(
                    vs=vs_,
                    noise=noise,
                    window=window,
                    scale=scale,
                    t=t,
                    n_u=n_u,
                    n_z=n_z,
                ).construct(t, y),
            )
        )
    if "gprv" in names:
        names = set(names) - {"gprv"}
        models.append(
            (
                "GP-RV",
                Vars(torch.float64),
                lambda vs_: (
                    GPRV(
                        vs=vs_,
                        noise=noise,
                        window=window,
                        scale=scale,
                        t=t,
                        n_u=n_u,
                        m_max=int(np.ceil(n_z / 2)),
                    ).construct(t, y)
                ),
            )
        )

    if len(names) > 0:
        names_str = ", ".join(f'"{name}"' for name in names)
        raise ValueError(f"Unknown names {names_str}.")

    return models


def train_models(train, models, wd=None, **kw_args):
    """Train models.

    Further takes in keyword arguments for :func:`.model.train`.

    Args:
        train (function): Training function.
        models (list): Models to train.
        wd (:class:`wbml.experiment.WorkingDirectory`, optional): Working
            directory to save samples to.

    Returns:
        list[:class:`stheno.Normal`]: Approximate posteriors.
    """
    dists = []

    for name, vs, construct_model in models:
        with wbml.out.Section(f"Training {name}"):
            construct_model(vs)
            dist = train(construct_model, vs, **kw_args)
            dists.append(dist)

    # Save results.
    if wd:
        wd.save([(q.mean, q.var) for q in dists], "dists.pickle")
        wd.save(
            [{name: vs[name] for name in vs.names} for _, vs, _ in models],
            "variables.pickle",
        )

    return dists


def analyse_models(
    models,
    dists,
    t,
    y,
    wd=None,
    t_plot=None,
    truth=None,
    true_kernel=None,
    true_noisy_kernel=None,
    comparative_kernel=None,
    x_range=None,
    y_range=None,
):
    """Analyse models.

    Args:
        models (list): Models.
        dists (list[:class:`stheno.Normal`]): Approximate posteriors.
        t (vector): Time points of data.
        y (vector): Observations.
        wd (:class:`wbml.experiment.WorkingDirectory`, optional): Working directory
            to save results to.
        t_plot (vector, optional): Time points to generate plots at. Defaults to `t`.
        truth (tuple[vector], optional): Tuple containing inputs and outputs
            associated to a truth.
        true_kernel (:class:`stheno.Kernel`, optional): True kernel that generated
            the data, not including noise.
        true_noisy_kernel (:class:`stheno.Kernel`, optional): True kernel that
            generated the data, including noise.
        comparative_kernel (function, optional): A function that takes in a
            variable container and gives back a kernel. A GP with this
            kernel will be trained on the data to compute a likelihood that
            will be compared to the ELBOs.
        x_range (dict, optional): Fix the x-range for plotting. Defaults to an empty
            dictionary.
        y_range (dict, optional): Fix the y-range for plotting. Defaults to an empty
            dictionary.
    """

    # Print the learned variables.
    with wbml.out.Section("Variables after optimisation"):
        for name, vs, construct_model in models:
            with wbml.out.Section(name):
                vs.print()

    analyse_elbos(
        models,
        dists,
        t=t,
        y=y,
        true_noisy_kernel=true_noisy_kernel,
        comparative_kernel=comparative_kernel,
    )
    analyse_plots(
        models,
        dists,
        t=t,
        y=y,
        wd=wd,
        true_kernel=true_kernel,
        t_plot=t_plot,
        truth=truth,
        x_range=x_range,
        y_range=y_range,
    )


def analyse_elbos(models, dists, t, y, true_noisy_kernel=None, comparative_kernel=None):
    """Compare ELBOs of models.

    Args:
        models (list): Models to train.
        dists (list[:class:`stheno.Normal`]): Approximate posteriors.
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
        wbml.out.kv("LML under true GP", GP(true_noisy_kernel)(t).logpdf(y))

    # Print LML under a trained GP if a comparative kernel is given.
    if comparative_kernel:

        def objective(vs_):
            gp = GP(sequential(comparative_kernel)(vs_))
            return -gp(t).logpdf(y)

        # Fit the GP.
        vs = Vars(torch.float64)
        lml_gp_opt = -minimise_l_bfgs_b(objective, vs, iters=1000)

        # Print likelihood.
        wbml.out.kv("LML under optimised GP", lml_gp_opt)

    # Estimate ELBOs.
    with wbml.out.Section("ELBOs"):
        for i, (name, vs, construct_model) in enumerate(models):
            model = construct_model(vs)
            wbml.out.kv(name, model.elbo(dists[i]))


def analyse_plots(
    models,
    dists,
    t,
    y,
    wd=None,
    true_kernel=None,
    t_plot=None,
    truth=None,
    x_range=None,
    y_range=None,
):
    """Analyse models in plots.

    Args:
        models (list): Models to train.
        dists (list[:class:`stheno.Normal`]): Approximate posteriors.
        t (vector): Time points of data.
        y (vector): Observations.
        wd (:class:`wbml.experiment.WorkingDirectory`, optional): Working directory
            to save the plots to.
        true_kernel (:class:`stheno.Kernel`, optional): True kernel that
            generates the data for comparison.
        t_plot (vector, optional): Time points to generate plots at. Defaults to `t`.
        truth (tuple[vector], optional): Tuple containing inputs and outputs
            associated to a truth.
        x_range (dict, optional): Fix the x-range for plotting. Defaults to an empty
            dictionary.
        y_range (dict, optional): Fix the y-range for plotting. Defaults to an empty
            dictionary.
    """
    # Set defaults.
    if t_plot is None:
        t_plot = t
    if x_range is None:
        x_range = {}
    if y_range is None:
        y_range = {}

    # Check whether `t` is roughly equally spaced. We allow small deviations.
    t_is_equally_spaced = max(np.abs(np.diff(np.diff(t)))) / max(np.abs(t)) < 5e-2

    # Plot predictions.
    plt.figure(figsize=(12, 8))

    for i, (name, vs, construct_model) in enumerate(models):
        # Construct model and make predictions.
        model = construct_model(vs)
        mu, std = model.predict(dists[i], t_plot)

        plt.subplot(3, 1, 1 + i)
        plt.title(f"Function ({name})")

        # Plot data.
        plt.scatter(t, y, c="black", label="Data")

        # Plot inducing models, if the model has them.
        if hasattr(model, "t_z"):
            plt.scatter(model.t_z, model.t_z * 0, s=5, marker="o", c="black")

        # Plot the predictions.
        plt.plot(t_plot, mu, c="tab:green", label="Prediction")
        plt.fill_between(t_plot, mu - std, mu + std, facecolor="tab:green", alpha=0.2)
        plt.fill_between(
            t_plot, mu - 2 * std, mu + 2 * std, facecolor="tab:green", alpha=0.2
        )
        error = 2 * B.sqrt(vs["noise"] + std ** 2)  # Model and noise error.
        plt.plot(t_plot, mu + error, c="tab:green", ls="--")
        plt.plot(t_plot, mu - error, c="tab:green", ls="--")

        # Plot true function.
        if truth:
            plt.plot(*truth, c="tab:red", label="Truth")

        # Set limit and format.
        if "function" in x_range:
            plt.xlim(*x_range["function"])
        else:
            plt.xlim(min(t_plot), max(t_plot))
        if "function" in y_range:
            plt.ylim(*y_range["function"])
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter("$%.1f$"))
        wbml.plot.tweak(legend=True)

    plt.tight_layout()
    if wd:
        plt.savefig(wd.file("prediction_function.pdf"))

    # Plot kernels.
    plt.figure(figsize=(12, 8))

    for i, (name, vs, construct_model) in enumerate(models):
        # Construct model and predict the kernel.
        model = construct_model(vs)
        with wbml.out.Section(f"Predicting kernel for {name}"):
            pred = model.predict_kernel(dists[i])

        # Compute true kernel.
        if true_kernel:
            k_true = true_kernel(pred.x).mat[0, :]

        # Estimate autocorrelation.
        if t_is_equally_spaced:
            t_ac = t - t[0]
            k_ac = autocorr(y, normalise=False)

        plt.subplot(3, 1, 1 + i)
        plt.title(f"Kernel ({name})")

        # Plot inducing points, if the model has them.
        plt.scatter(model.t_u, 0 * model.t_u, s=5, c="black")

        # Plot predictions.
        plt.plot(pred.x, pred.mean, c="tab:green", label="Prediction")
        plt.fill_between(
            pred.x,
            pred.err_68_lower,
            pred.err_68_upper,
            facecolor="tab:green",
            alpha=0.2,
        )
        plt.fill_between(
            pred.x,
            pred.err_95_lower,
            pred.err_95_upper,
            facecolor="tab:green",
            alpha=0.2,
        )
        plt.plot(pred.x, pred.samples, c="tab:red", lw=1)

        # Plot the true kernel.
        if true_kernel:
            plt.plot(pred.x, k_true, c="black", label="True", scaley=False)

        # Plot the autocorrelation of the data.
        if t_is_equally_spaced:
            plt.plot(t_ac, k_ac, c="tab:blue", label="Autocorrelation", scaley=False)

        # Set limits and format.
        if "kernel" in x_range:
            plt.xlim(*x_range["kernel"])
        else:
            plt.xlim(0, max(pred.x))
        if "kernel" in y_range:
            plt.ylim(*y_range["kernel"])
        wbml.plot.tweak(legend=True)

    plt.tight_layout()
    if wd:
        plt.savefig(wd.file("prediction_kernel.pdf"))

    # Plot PSDs.
    plt.figure(figsize=(12, 8))

    for i, (name, vs, construct_model) in enumerate(models):
        # Construct compute and predict PSD.
        model = construct_model(vs)
        with wbml.out.Section(f"Predicting PSD for {name}"):
            pred = model.predict_psd(dists[i])

        # Compute true PSD.
        if true_kernel:
            # TODO: Is `pred.x` okay, or should it be longer?
            freqs_true, psd_true = estimate_psd(
                pred.x, true_kernel(pred.x).mat[0, :], db=True
            )

        # Estimate PSD.
        if t_is_equally_spaced:
            t_ac = t - t[0]
            k_ac = autocorr(y, normalise=False)
            freqs_ac, psd_ac = estimate_psd(t_ac, k_ac, db=True)

        plt.subplot(3, 1, 1 + i)
        plt.title(f"PSD ({name})")

        # Plot predictions.
        plt.plot(pred.x, pred.mean, c="tab:green", label="Prediction")
        # TODO: `scalex` doesn't work with `fill_between`. Fix?
        plt.fill_between(
            pred.x,
            pred.err_68_lower,
            pred.err_68_upper,
            facecolor="tab:green",
            alpha=0.2,
        )
        plt.fill_between(
            pred.x,
            pred.err_95_lower,
            pred.err_95_upper,
            facecolor="tab:green",
            alpha=0.2,
        )
        plt.plot(pred.x, pred.samples, c="tab:red", lw=1)

        # Plot true PSD.
        if true_kernel:
            plt.plot(freqs_true, psd_true, c="black", label="True", scaley=False)

        # Plot PSD derived from the autocorrelation.
        if t_is_equally_spaced:
            plt.plot(
                freqs_ac, psd_ac, c="tab:blue", label="Autocorrelation", scaley=False
            )

        # Set limits and format.
        if "psd" in x_range:
            plt.xlim(*x_range["psd"])
        else:
            if t_is_equally_spaced:
                plt.xlim(0, max(freqs_ac))
            else:
                plt.xlim(0, max(pred.x))
        if "psd" in y_range:
            plt.ylim(*y_range["psd"])
        wbml.plot.tweak(legend=True)

    plt.tight_layout()
    if wd:
        plt.savefig(wd.file("prediction_psd.pdf"))

    # Plot Fourier features for GP-RV.
    plt.figure(figsize=(12, 8))

    for i, (name, vs, construct_model) in enumerate(models):
        # Construct model.
        model = construct_model(vs)

        # Predict Fourier features if it is a GP-RV.
        if isinstance(model, GPRV):
            mean, lower, upper = model.predict_fourier(dists[i])
        else:
            continue

        plt.subplot(3, 2, 1 + 2 * i)
        plt.title(f"Cosine Features ({name})")
        freqs = model.ms / (model.b - model.a)
        inds = np.concatenate(
            np.where(model.ms == 0) + np.where(model.ms <= model.m_max)
        )
        plt.errorbar(
            freqs[inds],
            mean[inds],
            (mean[inds] - lower[inds], upper[inds] - mean[inds]),
            ls="none",
            marker="o",
            capsize=3,
        )
        plt.xlabel("Frequency (Hz)")
        wbml.plot.tweak(legend=False)

        plt.subplot(3, 2, 2 + 2 * i)
        plt.title(f"Sine Features ({name})")
        freqs = np.maximum(model.ms - model.m_max, 0) / (model.b - model.a)
        inds = np.concatenate(
            np.where(model.ms == 0) + np.where(model.ms > model.m_max)
        )
        plt.errorbar(
            freqs[inds],
            mean[inds],
            (mean[inds] - lower[inds], upper[inds] - mean[inds]),
            ls="none",
            marker="o",
            capsize=3,
        )
        plt.xlabel("Frequency (Hz)")
        wbml.plot.tweak(legend=False)

    plt.tight_layout()
    if wd:
        plt.savefig(wd.file("prediction_fourier.pdf"))
