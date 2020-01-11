import warnings

import lab.tensorflow as B
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wbml.plot
import wbml.out
from gpcm.gprv import GPRV, determine_m_max
from gpcm.util import autocorr
from matplotlib.ticker import FormatStrFormatter
from matrix.util import ToDenseWarning
# noinspection PyUnresolvedReferences
from stheno import GP, EQ, Delta, Matern32, Matern12
from varz import Vars
from varz.tensorflow import minimise_l_bfgs_b

warnings.simplefilter(category=ToDenseWarning, action='ignore')

# Setup experiment.
n = 250
t = B.linspace(0, 40, n)
noise = 0.05**2
learn_noise = True
iters = 100

# Setup true model and GP-RV model.
vs = Vars(tf.float64)

# kernel = Matern32().stretch(1.5)*(lambda x: B.cos(2*B.pi*x/2))
# window = 4
# per = 1

kernel = EQ().stretch(0.5)
# kernel = Matern32().stretch(0.5)
# kernel = Matern12().stretch(0.75)
window = 1
per = 0.5

# Sample data.
gp = GP(kernel + noise*Delta())
y = B.flatten(gp(t).sample())


def construct_model(vs):
    model = GPRV(vs,
                 noise=noise, window=window, t=t,
                 n_u=40, m_max=determine_m_max(per=per, t=t))
    model.construct(t, y)
    return model


# Perform optimisation and print variables before and after.
construct_model(vs)
vs.print()
objective = tf.function(lambda vs_: -construct_model(vs_).elbo(),
                        autograph=False)
minimise_l_bfgs_b(objective, vs, iters=iters, trace=True,
                  names=list(set(vs.names) - {'noise'}))
if learn_noise:
    minimise_l_bfgs_b(objective, vs, iters=iters, trace=True)
vs.print()

model = construct_model(vs)

# Print ELBO versus LML of true GP.
wbml.out.kv('ELBO', model.elbo())
wbml.out.kv('LML of true GP', gp(t).logpdf(y))

# Plot predictions.
mu, std = model.predict()
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1).set_title('Function')
plt.scatter(t, y, c='black', label='Data')
plt.plot(t, mu, c='tab:green', label='Prediction')
plt.fill_between(t, mu - std, mu + std,
                 facecolor='tab:green', alpha=0.15)
plt.fill_between(t, mu - 2*std, mu + 2*std,
                 facecolor='tab:green', alpha=0.15)
plt.fill_between(t, mu - 3*std, mu + 3*std,
                 facecolor='tab:green', alpha=0.15)
plt.plot(t, mu + 3*std + vs['noise']**.5, c='tab:green', ls='--')
plt.plot(t, mu - 3*std - vs['noise']**.5, c='tab:green', ls='--')
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%.1f$'))
wbml.plot.tweak(legend=True)

# Plot kernel.
pred = model.predict_kernel()
plt.subplot(2, 3, 4).set_title('Kernel')
plt.plot(pred.t, pred.mean, c='tab:green', label='Prediction')
plt.plot(pred.t, kernel(pred.t).mat[0, :], c='black', label='True')
plt.fill_between(pred.t, pred.err_68[0], pred.err_68[1],
                 facecolor='tab:green', alpha=0.15)
plt.fill_between(pred.t, pred.err_95[0], pred.err_95[1],
                 facecolor='tab:green', alpha=0.15)
plt.fill_between(pred.t, pred.err_99[0], pred.err_99[1],
                 facecolor='tab:green', alpha=0.15)
dt = model.t_u[1] - model.t_u[0]
lags = int(np.ceil(max(pred.t/dt)))
plt.plot(np.arange(lags + 1)*dt, autocorr(y, lags),
         c='tab:blue', label='Autocorrelation')
plt.plot(pred.t, pred.samples, c='tab:red', lw=1)
plt.scatter(model.t_u, 0*model.t_u, s=5, c='black')
wbml.plot.tweak(legend=True)

# Plot Fourier features.
mean, lower, upper = model.predict_fourier()
plt.subplot(2, 3, 5).set_title('Cosine Features')
freqs = model.ms/B.to_numpy(model.b - model.a)
inds = np.concatenate(np.where(model.ms == 0) +
                      np.where(model.ms <= model.m_max))
plt.errorbar(freqs[inds], mean[inds], (mean[inds] - lower[inds],
                                       upper[inds] - mean[inds]),
             ls='none', marker='o', capsize=3)
plt.xlabel('Frequency (Hz)')
wbml.plot.tweak()

plt.subplot(2, 3, 6).set_title('Sine Features')
freqs = np.maximum(model.ms - model.m_max, 0)/B.to_numpy(model.b - model.a)
inds = np.concatenate(np.where(model.ms == 0) +
                      np.where(model.ms > model.m_max))
plt.errorbar(freqs[inds], mean[inds], (mean[inds] - lower[inds],
                                       upper[inds] - mean[inds]),
             ls='none', marker='o', capsize=3)
plt.xlabel('Frequency (Hz)')
wbml.plot.tweak()

plt.tight_layout()
plt.show()
