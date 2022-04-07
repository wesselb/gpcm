import matplotlib.pyplot as plt
import numpy as np
import wbml.metric as metric
from wbml.experiment import WorkingDirectory
from wbml.plot import tex, tweak, pdfcrop

tex()


def format_num(x):
    if np.isnan(x):
        return "nan"

    # Get base and exponent of number.
    base, exp = "{:.2e}".format(x).split("e")

    # Split off sign.
    exp_sign, exp = exp[0], exp[1:]

    # Remove leading zeros.
    while len(exp) > 0 and exp[0] in {"0"}:
        exp = exp[1:]

    # Simplify in a couple cases
    if exp_sign == "-" and exp == "2":
        base = f"{float(base) / 100:.2f}"
        exp = ""
    elif exp_sign == "-" and exp == "1":
        base = f"{float(base) / 10:.2f}"
        exp = ""
    elif exp_sign == "+" and exp == "1":
        base = f"{float(base) * 10:.1f}"
        exp = ""
    elif exp_sign == "+" and exp == "2":
        base = f"{float(base) * 100:.0f}"
        exp = ""
    elif exp_sign == "-":
        # Remove one level of significance for spacing.
        base = base[:-1]

    # Format minus sign of base.
    if base[0] == "-":
        base = "\\text{-}" + base[1:]
    else:
        base = "\\hphantom{\\text{-}}" + base

    # Format exponent.
    if exp_sign == "-" and len(exp) > 0:
        exp = "\\,\\text{-}" + exp
    else:
        exp = "\\,\\hphantom{\\text{-}}" + exp
    if exp != "\\,\\hphantom{\\text{-}}":
        exp = f"\\text{{\\textsc{{e}}}}{{{exp}}}"
    else:
        exp = ""

    return f"{base}{exp}"


def print_estimates(estimates, show_error=False):
    out = ""
    for estimate in estimates:
        if isinstance(estimate, tuple):
            value, error = estimate
        else:
            value = estimate
            error = 0
        out += f"& ${format_num(value)} "
        if error > 0 and show_error:
            out += f"{{ \\scriptstyle \\,\\pm\\, {format_num(error)} }}$ "
        else:
            out = out[:-1] + "$ "
    return out[2:]


wd_out = WorkingDirectory("_experiments", "comparison")
wd = WorkingDirectory(
    "server",
    "_experiments",
    "comparison",
    observe=True,
)


def kernel_analysis(data_name, model, mode, until=4):
    k = wd.load(data_name, "data.pickle")["k"]
    t, mean1, var1 = wd.load(data_name, "structured", model, "k_pred.pickle")
    elbo1 = wd.load(data_name, "structured", model, "elbo.pickle")
    t, mean2, var2 = wd.load(data_name, "mean-field", model, "k_pred.pickle")
    elbo2 = wd.load(data_name, "mean-field", model, "elbo.pickle")
    if mode == "mean-field-elbo":
        return elbo2
    elif mode == "structured-elbo":
        return elbo1
    elif mode == "mean-field-mll":
        return metric.smll(mean2[t <= until], var2[t <= until], k[t <= until])
    elif mode == "mean-field-rmse":
        return metric.rmse(mean2[t <= until], k[t <= until])
    elif mode == "structured-mll":
        return metric.smll(mean1[t <= until], var1[t <= until], k[t <= until])
    elif mode == "structured-rmse":
        return metric.rmse(mean1[t <= until], k[t <= until])
    else:
        raise ValueError(f'Bad mode "{mode}".')


def gp_logpdf(data_name):
    return wd.load(data_name, "data.pickle")["true_logpdf"]


out = "\\toprule \n"
out += (
    "\\textsc{Model} "
    " & \\textsc{Data} "
    "& \\multicolumn{2}{c}{\\textsc{MLL}} "
    "& \\multicolumn{2}{c}{\\textsc{RMSE}} "
    "\\\\ \n"
    " & "
    " & \\textsc{MF} & \\textsc{S}"
    " & \\textsc{MF} & \\textsc{S}"
    "\\\\ "
    "\\midrule \n"
)
out += "\\textsc{GPCM}"
out += " & \\textsc{EQ} & "
estimates = [
    kernel_analysis("eq", "gpcm", mode="mean-field-mll"),
    kernel_analysis("eq", "gpcm", mode="structured-mll"),
    kernel_analysis("eq", "gpcm", mode="mean-field-rmse"),
    kernel_analysis("eq", "gpcm", mode="structured-rmse"),
]
out += print_estimates(estimates) + "\\\\ \n"

out += "\\textsc{CGPCM}"
out += " & \\textsc{CEQ} & "
estimates = [
    kernel_analysis("ceq-1", "cgpcm", mode="mean-field-mll"),
    kernel_analysis("ceq-1", "cgpcm", mode="structured-mll"),
    kernel_analysis("ceq-1", "cgpcm", mode="mean-field-rmse"),
    kernel_analysis("ceq-1", "cgpcm", mode="structured-rmse"),
]
out += print_estimates(estimates) + "\\\\ \n"

out += "\\textsc{GPRVM}"
out += " & \\textsc{Matern–$\\frac{1}{2}$} & "
estimates = [
    kernel_analysis("matern12", "gprvm", mode="mean-field-mll"),
    kernel_analysis("matern12", "gprvm", mode="structured-mll"),
    kernel_analysis("matern12", "gprvm", mode="mean-field-rmse"),
    kernel_analysis("matern12", "gprvm", mode="structured-rmse"),
]
out += print_estimates(estimates) + "\\\\ \n"

out += "\\bottomrule \n"
print(out)


def plot_kernel_predictions(wd, model, data_name, legend=True, first=False):
    k = wd.load(data_name, "data.pickle")["k"]
    t, mean1, var1 = wd.load(data_name, "structured", model, "k_pred.pickle")
    t, mean2, var2 = wd.load(data_name, "mean-field", model, "k_pred.pickle")
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
plot_kernel_predictions(wd, "gpcm", "eq", legend=False, first=True)
plt.subplot(1, 3, 2)
plt.title("CGPCM on CEQ")
plot_kernel_predictions(wd, "cgpcm", "ceq-1", legend=False)
plt.subplot(1, 3, 3)
plt.title("RGPCM on Matern–$\\frac{1}{2}$")
plot_kernel_predictions(wd, "gprvm", "matern12")
plt.savefig(wd_out.file("comparison.pdf"))
pdfcrop(wd_out.file("comparison.pdf"))
plt.show()