import lab as B
from gpcm import GPCM, CGPCM, GPRVM
from slugify import slugify
from stheno import EQ, CEQ, Exp, GP, Delta
from wbml.experiment import WorkingDirectory
import sys
import wbml.out as out

out.report_time = True

seed = sys.argv[1]
wd = WorkingDirectory("_experiments", "kernels", seed)

B.epsilon = 1e-8

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

    for scheme, fit_args in [
        ("mean-field-ca", {}),
        ("structured", {"optimise_hypers": False}),
    ]:
        for model in [
            GPCM(
                scheme=scheme,
                window=window,
                scale=scale,
                noise=noise,
                n_u=n_u,
                n_z=n_z,
                t=t,
            ),
            CGPCM(
                scheme=scheme,
                window=window,
                scale=scale,
                noise=noise,
                n_u=n_u,
                n_z=n_z,
                t=t,
            ),
            GPRVM(
                scheme=scheme,
                window=window,
                scale=scale,
                noise=noise,
                n_u=n_u,
                m_max=n_z // 2,
                t=t,
            ),
        ]:
            prefix = (slugify(str(kernel)), scheme, slugify(model.name))

            # Fit model and predict function and kernel.
            model.fit(t, y, **fit_args)
            elbo = model.elbo(t, y)
            posterior = model.condition(t, y)
            f_pred = posterior.predict(t)
            k_pred = posterior.predict_kernel(t_k)

            # Save stuff.
            model.save(wd.file(*prefix, "model.pickle"))
            wd.save(elbo, *prefix, "elbo.pickle")
            wd.save((t,) + f_pred, *prefix, "f_pred.pickle")
            wd.save((k_pred.x, k_pred.mean, k_pred.var), *prefix, "k_pred.pickle")
