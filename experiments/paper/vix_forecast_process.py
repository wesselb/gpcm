import numpy as np
import scipy.stats as st
import wbml.metric as metric
import wbml.out as out
from wbml.experiment import WorkingDirectory

wd = WorkingDirectory("_experiments", "vix_forecast", observe=True)


def compute_metrics(model, summarise=True):
    rmses, mlls = [], []
    preds = wd.load(model, "preds.pickle")
    for (y, mean, var) in preds:
        rmses.append(metric.rmse(mean, y))
        mlls.append(metric.mll(mean, var, y))
    if summarise:
        with out.Section(model.upper()):
            for name, values in [("MLL", mlls), ("RMSE", rmses)]:
                with out.Section(name):
                    out.kv("Mean", np.mean(values))
                    out.kv("Error", 1.96 * np.std(values) / len(values) ** 0.5)
    else:
        return mlls, rmses


def compare(model1, model2):
    mlls1, rmses1 = compute_metrics(model1, summarise=False)
    mlls2, rmses2 = compute_metrics(model2, summarise=False)
    diff_rmses = [x - y for x, y in zip(rmses1, rmses2)]
    diff_mlls = [x - y for x, y in zip(mlls1, mlls2)]
    with out.Section(f"{model1.upper()} - {model2.upper()}"):
        for name, values1, values2 in [("MLL", mlls1, mlls2), ("RMSE", rmses1, rmses2)]:
            diff = [x - y for x, y in zip(values1, values2)]
            with out.Section(name):
                mean = np.mean(diff)
                error = 1.96 * np.std(diff_mlls) / len(diff_mlls) ** 0.5
                out.kv("Mean", mean)
                out.kv("Error", error)
                out.kv("p-value", st.norm.cdf(-abs(mean) / (error / 1.96)))


compute_metrics("gpcm")
compute_metrics("cgpcm")
compute_metrics("rgpcm")
compare("rgpcm", "gpcm")
compare("rgpcm", "cgpcm")
compare("cgpcm", "gpcm")
