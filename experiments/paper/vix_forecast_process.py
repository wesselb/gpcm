import numpy as np
import scipy.stats as st
import wbml.metric as metric
import wbml.out as out
from wbml.experiment import WorkingDirectory

# Setup script.
wd = WorkingDirectory("_experiments", "vix_forecast_process")
wd_results = WorkingDirectory("_experiments", "vix_forecast", observe=True)


def compute_metrics(model, summarise=True):
    """Compute metrics.

    Args:
        model (str): Name of the model folder.
        summarise (bool, optional): Summarise the metrics rather than given the data
            back. Defaults to `True`.

    Returns:
        union[None, tuple[:class:`np.array`, :class:`np.array`]]: The metrics if
            `summarise` is `False`. Otherwise nothing.
    """
    rmses, mlls = [], []
    preds = wd_results.load(model, "preds.pickle")
    for (y, mean, var) in preds:
        rmses.append(metric.rmse(mean, y))
        mlls.append(metric.mll(mean, var, y))
    if summarise:
        with out.Section(model.upper()):
            for name, values in [("MLL", mlls), ("RMSE", rmses)]:
                with out.Section(name):
                    out.kv("Mean", np.mean(values))
                    out.kv("Std", np.std(values) / len(values) ** 0.5)
    else:
        return mlls, rmses


def compare(model1, model2):
    """Compare two models.

    Args:
        model1 (str): Model folder of the first model to compare.
        model2 (str): Model folder of the second model to compare.
    """
    mlls1, rmses1 = compute_metrics(model1, summarise=False)
    mlls2, rmses2 = compute_metrics(model2, summarise=False)
    with out.Section(f"{model1.upper()} - {model2.upper()}"):
        for name, values1, values2 in [("MLL", mlls1, mlls2), ("RMSE", rmses1, rmses2)]:
            diff = [x - y for x, y in zip(values1, values2)]
            with out.Section(name):
                mean = np.mean(diff)
                std = np.std(diff) / len(diff) ** 0.5
                out.kv("Mean", mean)
                out.kv("Std", std)
                out.kv("p-value", st.norm.cdf(-abs(mean) / std))


compute_metrics("rgpcm")
compute_metrics("cgpcm")
compute_metrics("gpcm")
compare("rgpcm", "gpcm")
compare("rgpcm", "cgpcm")
compare("cgpcm", "gpcm")
