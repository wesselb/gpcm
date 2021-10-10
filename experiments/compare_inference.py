import time

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.out as out
from stheno import EQ, GP
from varz import minimise_l_bfgs_b
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak, pdfcrop

from gpcm import GPCM
from gpcm.approx import _fit_mean_field_ca

out.report_time = True

wd = WorkingDirectory("_experiments", "compare_inference")

B.epsilon = 1e-8

# Setup experiment.
noise = 0.5
t = B.linspace(0, 20, 500)

# Setup GPCM models.
window = 2
scale = 1
n_u = 40
n_z = 40

# Sample data.
kernel = EQ()
y = B.flatten(GP(kernel)(t, noise).sample())
gp_logpdf = GP(kernel)(t, noise).logpdf(y)


class Tracker:
    def __init__(self):
        self.ref = time.time()
        self.times = []
        self.values = []

    def track(self, value):
        if hasattr(value, "primal"):
            value = value.primal
        self.times.append(time.time() - self.ref)
        self.values.append(value)

    def pause(self):
        class Context:
            def __enter__(ctx):
                ctx.start = time.time()

            def __exit__(ctx, exc_type, exc_val, exc_tb):
                elapsed = time.time() - ctx.start
                self.ref += elapsed

        return Context()

    def get_xy(self, start=2):
        x = np.array(self.times[start:]) - self.times[start]
        y = np.array(self.values[start:])
        return x, y


def jit(vs, objective):
    # Run once to ensure that all parameters exist.
    objective(vs)

    @B.jit
    def objective_vectorised(params, *args):
        vs_copy = vs.copy()
        vs_copy.set_latent_vector(params)
        return objective(vs_copy, *args)

    def objective_wrapped(vs_, *args):
        return objective_vectorised(vs_.get_latent_vector(), *args)

    return objective_wrapped


# MF

model = GPCM(
    scheme="mean-field",
    window=window,
    scale=scale,
    noise=noise,
    n_u=n_u,
    n_z=n_z,
    t=t,
)
# Save initialisation and apply to next models for fair comparison.
instance = model()
init_q_u = instance.approximation.q_u
init_q_z = instance.approximation.q_z


def objective_raw(vs_):
    _, elbo = model(vs_).approximation.elbo(B.global_random_state(vs_.dtype), t, y)
    return -elbo


objective_jitted = jit(model.vs, objective_raw)
tracker_mf = Tracker()


def objective(vs_):
    nelbo = objective_jitted(vs_)
    tracker_mf.track(-nelbo)
    return nelbo


minimise_l_bfgs_b(
    objective,
    model.vs,
    trace=True,
    iters=200,
    jit=False,
)


# Collapsed MF

model = GPCM(
    scheme="mean-field",
    window=window,
    scale=scale,
    noise=noise,
    n_u=n_u,
    n_z=n_z,
    t=t,
)
instance = model()
instance.approximation.q_u = init_q_u
instance.approximation.q_z = init_q_z


def objective_raw(vs_):
    _, elbo = model(vs_).approximation.elbo(
        B.global_random_state(vs_.dtype),
        t,
        y,
        collapsed="z",
    )
    return -elbo


objective_jitted = jit(model.vs, objective_raw)
tracker_cmf = Tracker()


def objective(vs_):
    nelbo = objective_jitted(vs_)
    tracker_cmf.track(-nelbo)
    return nelbo


minimise_l_bfgs_b(
    objective,
    model.vs,
    trace=True,
    iters=200,
    jit=False,
)


# CA

model = GPCM(
    scheme="mean-field",
    window=window,
    scale=scale,
    noise=noise,
    n_u=n_u,
    n_z=n_z,
    t=t,
)
instance = model()
instance.approximation.q_u = init_q_u
instance.approximation.q_z = init_q_z

tracker_ca = Tracker()
count = 1


def callback(q_u, q_z):
    global count
    count += 1
    if count < 20 or count % 5 == 0:
        with tracker_ca.pause():
            instance = model()
            instance.approximation.q_u = q_u
            instance.approximation.q_z = q_z

            _, elbo_ca = instance.approximation.elbo(
                B.global_random_state(instance.dtype), t, y
            )
        tracker_ca.track(elbo_ca)


_fit_mean_field_ca(model(), t, y, callback=callback)

# Structured

model = GPCM(
    scheme="structured",
    window=window,
    scale=scale,
    noise=noise,
    n_u=n_u,
    n_z=n_z,
    t=t,
)
instance = model()
instance.approximation.q_u = init_q_u
instance.approximation.q_z = init_q_z

tracker_s = Tracker()
count = 1


def callback(q_u, q_z):
    global count
    count += 1
    if count < 20 or count % 5 == 0:
        with tracker_s.pause():
            instance = model()
            instance.approximation.q_u = q_u

            state = B.global_random_state(instance.dtype)
            state, elbo_s = instance.approximation.elbo_collapsed_z(
                state,
                t,
                y,
                num_samples=100,
            )
            B.set_global_random_state(state)
        tracker_s.track(elbo_s)


_fit_mean_field_ca(model(), t, y, callback=callback)


# Plot result

t_mf, elbo_mf = tracker_mf.get_xy(start=2)
t_cmf, elbo_cmf = tracker_cmf.get_xy(start=2)
t_ca, elbo_ca = tracker_ca.get_xy(start=0)
t_s, elbo_s = tracker_s.get_xy(start=0)

assert t_mf[1] < 1
assert t_cmf[1] < 1
assert t_ca[1] < 1
assert t_s[1] < 1

plt.figure(figsize=(5, 4))
plt.axhline(y=gp_logpdf, ls="--", c="black", lw=1, label="GP")
plt.plot(
    # These times should line up exactly, but they might not.
    t_s / max(t_s) * max(t_ca),
    elbo_s,
    label="Structured",
)
plt.plot(
    t_ca,
    elbo_ca,
    label="CA",
)
plt.plot(
    t_cmf,
    np.maximum.accumulate(elbo_cmf),
    label="Collapsed MF",
)
plt.plot(
    t_mf,
    np.maximum.accumulate(elbo_mf),
    label="MF",
)
plt.xlabel("Time (s)")
plt.ylabel("ELBO")
plt.ylim(-900, -550)
plt.xlim(0, 30)
tweak(legend_loc="lower right")
plt.savefig(wd.file("elbos.pdf"))
pdfcrop(wd.file("elbos.pdf"))
plt.show()
