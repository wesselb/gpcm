import lab as B
import wbml.out as out
from gpcm import GPCM, CGPCM, RGPCM
from slugify import slugify
from stheno import EQ, CEQ, Exp, GP, Delta
from wbml.experiment import WorkingDirectory

# Setup script.
out.report_time = True
B.epsilon = 1e-8
wd = WorkingDirectory("_experiments", "comparison")

# Setup experiment.
noise = 1.0
t = B.linspace(0, 40, 400)
t_k = B.linspace(0, 4, 200)

# Setup GPCM models.
window = 2
scale = 0.5
n_u = 30
n_z = 80

for kernel, model in [
    (
        EQ(),
        lambda scheme: GPCM(
            scheme=scheme,
            window=window,
            scale=scale,
            noise=noise,
            n_u=n_u,
            n_z=n_z,
            t=t,
        ),
    ),
    (
        CEQ(1),
        lambda scheme: CGPCM(
            scheme=scheme,
            window=window,
            scale=scale,
            noise=noise,
            n_u=n_u,
            n_z=n_z,
            t=t,
        ),
    ),
    (
        Exp(),
        lambda scheme: RGPCM(
            scheme=scheme,
            window=window,
            scale=scale,
            noise=noise,
            n_u=n_u,
            m_max=n_z // 2,
            t=t,
        ),
    ),
]:
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

    for scheme in ["mean-field", "structured"]:
        prefix = (slugify(str(kernel)), scheme, slugify(model.name))

        # Fit model and predict function and kernel.
        model.fit(t, y, iters=20_000)
        elbo = model.elbo(t, y)
        posterior = model.condition(t, y)
        f_pred = posterior.predict(t)
        k_pred = posterior.predict_kernel(t_k)

        # Save stuff.
        model.save(wd.file(*prefix, "model.pickle"))
        wd.save(elbo, *prefix, "elbo.pickle")
        wd.save((t,) + f_pred, *prefix, "f_pred.pickle")
        wd.save((k_pred.x, k_pred.mean, k_pred.var), *prefix, "k_pred.pickle")
