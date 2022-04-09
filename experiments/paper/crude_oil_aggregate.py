import numpy as np
import wbml.metric as metric
import wbml.out as out
from scipy.stats import ttest_rel
from wbml.experiment import WorkingDirectory

# Load all experiments and compute metrics.
names = ["GPCM", "CGPCM", "RGPCM"]
mlls = {name: [] for name in names}
rmses = {name: [] for name in names}
for year in range(2012, 2017 + 1):
    wd = WorkingDirectory("_experiments", "crude_oil", str(year), observe=True)
    t, y = wd.load("data.pickle")["test"]
    for name in names:
        _, mean, var = wd.load(name.lower(), "pred_f_test.pickle")
        mlls[name].append(metric.mll(mean, var, y))
        rmses[name].append(metric.rmse(mean, y))

# Print aggregate results.
for name in names:
    with out.Section(name):
        out.kv("MLL", np.mean(mlls[name]))
        out.kv("MLL (std)", np.std(mlls[name]) / len(mlls[name]) ** 0.5)
        out.kv("RMSE", np.mean(rmses[name]))
        out.kv("RMSE (std)", np.std(rmses[name]) / len(rmses[name]) ** 0.5)

# Compare results.
for name1, name2 in [("RGPCM", "CGPCM"), ("RGPCM", "GPCM"), ("CGPCM", "GPCM")]:
    with out.Section(f"{name1} - {name2}"):
        out.kv("MLL", np.mean(mlls[name1]) - np.mean(mlls[name2]))
        out.kv("MLL (p)", ttest_rel(mlls[name1], mlls[name2]).pvalue)
        out.kv("RMSE", np.mean(rmses[name1]) - np.mean(rmses[name2]))
        out.kv("RMSE (p)", ttest_rel(rmses[name1], rmses[name2]).pvalue)
