import lab as B
from gpcm import GPCM, CGPCM, GPRVM
from slugify import slugify
from stheno import EQ, CEQ, Exp, GP, Delta
from wbml.experiment import WorkingDirectory
import wbml.out as out

out.report_time = True

wd = WorkingDirectory("_experiments", "kernels")

# Setup experiment.
noise = 0.2
t = B.linspace(0, 20, 100)
t_k = B.linspace(0, 4, 200)

# Setup GPCM models.
window = 2
scale = 1
n_u = 30
n_z = 80

for kernel in [EQ(), CEQ(1), Exp()]:
    # Sample data.
    gp_f = GP(kernel)
    gp_y = gp_f + GP(noise * Delta(), measure=gp_f.measure)
    f, y = map(B.flatten, gp_f.measure.sample(gp_f(t), gp_y(t)))
    wd.save(
        {
            "t": t,
            "f": f,
            "k": B.flatten(kernel(t_k, 0)),
            "y": y,
            "true_logpdf": gp_y(t).logpdf(y),
        },
        slugify(str(kernel)),
        "data.pickle",
    )

    for scheme in ["mean-field-ca", "structured"]:
        for model in [
            GPCM(scheme=scheme, window=window, scale=scale, n_u=n_u, n_z=n_z),
            CGPCM(scheme=scheme, window=window, scale=scale, n_u=n_u, n_z=n_z),
            GPRVM(scheme=scheme, window=window, scale=scale, n_u=n_u, m_max=n_z // 2),
        ]:
            prefix = (slugify(str(kernel)), scheme, slugify(model.name))

            # Setup fit arguments.
            if scheme == "structured":
                fit_kw_args = {"optimise_hypers": False}
            else:
                fit_kw_args = {}

            # Fit and predict model.
            model.fit(t, y, **fit_kw_args)
            elbo = model.elbo(t, y)
            posterior = model.condition(t, y)
            f_pred = model.predict(t)
            k_pred = model.predict_kernel(t)

            # Save stuff.
            model.save(wd.file(*prefix, "model.pickle"))
            wd.save(elbo, *prefix, "elbo.pickle")
            wd.save((f_pred.x, f_pred.mean, f_pred.var), *prefix, "f_pred.pickle")
            wd.save((k_pred.x, k_pred.mean, k_pred.var), *prefix, "k_pred.pickle")
