import matplotlib.pyplot as plt
import numpy as np
import wbml.out as out
from wbml.experiment import WorkingDirectory
from wbml.metric import smll, rmse
from wbml.plot import tex, tweak, pdfcrop

# Setup script.
tex()
wd = WorkingDirectory("_experiments", "comparison_process")
wd_results = WorkingDirectory("_experiments", "comparison", observe=True)


def kernel_analysis(data, scheme, model, metric, until=4):
    """Analyse the prediction for a kernel."""
    k = wd_results.load(data, "data.pickle")["k"]
    t, mean, var = wd_results.load(data, scheme, model, "k_pred.pickle")
    inds = t <= until
    if metric == "smll":
        return smll(mean[inds], var[inds], k[inds])
    elif metric == "rmse":
        return rmse(mean[inds], k[inds])
    else:
        raise ValueError(f'Bad metric "{metric}".')


for model, kernel in [("gpcm", "eq"), ("cgpcm", "ceq-1"), ("rgpcm", "matern12")]:
    with out.Section(model.upper()):
        with out.Section("MLL"):
            out.kv("MF", kernel_analysis(kernel, "mean-field", model, "smll"))
            out.kv("MF", kernel_analysis(kernel, "structured", model, "smll"))
        with out.Section("RMSE"):
            out.kv("MF", kernel_analysis(kernel, "mean-field", model, "rmse"))
            out.kv("MF", kernel_analysis(kernel, "structured", model, "rmse"))


def plot_kernel_predictions(model, data_name, legend=True, first=False):
    """Plot the prediction for a kernel."""
    k = wd_results.load(data_name, "data.pickle")["k"]
    t, mean1, var1 = wd_results.load(data_name, "structured", model, "k_pred.pickle")
    t, mean2, var2 = wd_results.load(data_name, "mean-field", model, "k_pred.pickle")
    plt.plot(t, k, label="Truth", style="train")
    plt.plot(t, mean1, label="Structured", style="pred")
    plt.fill_between(
        t,
        mean1 - 1.96 * np.sqrt(var1),
        mean1 + 1.96 * np.sqrt(var1),
        style="pred",
    )
    plt.plot(t, mean1 + 1.96 * np.sqrt(var1), style="pred", lw=1)
    plt.plot(t, mean1 - 1.96 * np.sqrt(var1), style="pred", lw=1)
    plt.plot(t, mean2, label="Mean-field", style="pred2")
    plt.fill_between(
        t,
        mean2 - 1.96 * np.sqrt(var2),
        mean2 + 1.96 * np.sqrt(var2),
        style="pred2",
    )
    plt.plot(t, mean2 + 1.96 * np.sqrt(var2), style="pred2", lw=1)
    plt.plot(t, mean2 - 1.96 * np.sqrt(var2), style="pred2", lw=1)
    plt.yticks([0, 0.5, 1])
    plt.xticks([0, 2, 4])
    plt.xlim(0, 4)
    plt.ylim(-0.25, 1.25)
    if not first:
        plt.gca().set_yticklabels([])
    tweak(legend=legend)


plt.figure(figsize=(7.5, 3))
plt.subplot(1, 3, 1)
plt.title("GPCM on EQ")
plot_kernel_predictions("gpcm", "eq", legend=False, first=True)
plt.subplot(1, 3, 2)
plt.title("CGPCM on CEQ")
plot_kernel_predictions("cgpcm", "ceq-1", legend=False)
plt.subplot(1, 3, 3)
plt.title("RGPCM on Matern–$\\frac{1}{2}$")
plot_kernel_predictions("rgpcm", "matern12")
plt.savefig(wd.file("comparison.pdf"))
pdfcrop(wd.file("comparison.pdf"))
plt.show()
