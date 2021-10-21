import numpy as np
import wbml.metric as metric
from wbml.experiment import WorkingDirectory

wd = WorkingDirectory("server", "_experiments", "vix_forecast", observe=True)


def compute_metrics(model, summarise=True):
    rmses, mlls = [], []
    preds = wd.load(model, "preds.pickle")
    for (y, mean, var) in preds:
        rmses.append(metric.rmse(mean, y))
        mlls.append(metric.mll(mean, var, y))
    if summarise:
        return (
            (np.mean(mlls), np.std(mlls) / len(mlls) ** 0.5),
            (np.mean(rmses), np.std(rmses) / len(rmses) ** 0.5),
        )
    else:
        return mlls, rmses


def compare(model1, model2):
    mlls1, rmses1 = compute_metrics(model1, summarise=False)
    mlls2, rmses2 = compute_metrics(model2, summarise=False)
    diff_rmses = [x - y for x, y in zip(rmses1, rmses2)]
    diff_mlls = [x - y for x, y in zip(mlls1, mlls2)]
    return (
        np.mean(diff_mlls) / (np.std(diff_mlls) / len(diff_mlls) ** 0.5),
        np.mean(diff_rmses) / (np.std(diff_rmses) / len(diff_rmses) ** 0.5),
    )


print("GPRVM", compute_metrics("gprvm"))
print(compare("gprvm", "gpcm"))
print(compare("gprvm", "cgpcm"))
print("GPCM", compute_metrics("gpcm"))
print(compare("gpcm", "cgpcm"))
print(compare("gpcm", "gprvm"))
print("CGPCM", compute_metrics("cgpcm"))
print(compare("cgpcm", "gpcm"))
print(compare("cgpcm", "gprvm"))
