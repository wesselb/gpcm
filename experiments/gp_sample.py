import warnings

import lab.tensorflow as B
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wbml.plot
from gpcm.elbo import construct, elbo, predict, predict_fourier, predict_kernel
from gpcm.gprv import GPRV
from gpcm.util import autocorr, pd_inv
from matplotlib.ticker import FormatStrFormatter
from matrix.util import ToDenseWarning
# noinspection PyUnresolvedReferences
from stheno import GP, EQ, Delta, Matern32, Matern12
from varz import Vars
from varz.tensorflow import minimise_l_bfgs_b

warnings.simplefilter(category=ToDenseWarning, action='ignore')
B.epsilon = 1e-10

# Setup experiment.
n = 250
t = B.linspace(0, 40, n)
noise = 0.05**2
learn_noise = False
iters = 200

# Setup true model and GP-RV model.
# kernel = Matern32().stretch(1.5)*(lambda x: B.cos(2*B.pi*x/2))
# model = GPRV(window=4, per=1, t=t)
kernel = EQ().stretch(0.5)
# kernel = Matern32().stretch(0.5)
# kernel = Matern12().stretch(0.75)
model = GPRV(window=1, per=0.5, t=t)

# Sample data.
y = B.flatten(GP(kernel + noise*Delta())(t).sample())

# Initialise variables.
vs = Vars(tf.float64)
vs.bounded(model.alpha_t, name='alpha_t')
vs.bounded(model.lam, name='lambda')
vs.bounded(model.gamma, name='gamma')
vs.bounded(model.gamma_t, name='gamma_t')
vs.bounded(noise**.5, lower=0.1*noise**.5, upper=1, name='sigma')
vs.unbounded(B.ones(model.n_u, 1), name='mu_u')
vs.positive_definite(B.dense(pd_inv(model.K_u())), name='cov_u')


def build(vs):
    return construct(GPRV(lam=vs['lambda'],
                          alpha=model.alpha,
                          alpha_t=vs['alpha_t'],
                          gamma=vs['gamma'],
                          gamma_t=vs['gamma_t'],
                          a=model.a,
                          b=model.b,
                          m_max=model.m_max,
                          ms=model.ms,
                          t_u=model.t_u),
                     t, y, vs['sigma'], vs['mu_u'], vs['cov_u'])


# Perform optimisation and print variables before and after.
vs.print()
objective = tf.function(lambda vs_: -elbo(build(vs_)), autograph=False)
minimise_l_bfgs_b(objective, vs, iters=iters, trace=True,
                  names=list(set(vs.names) - {'sigma'}))
if learn_noise:
    minimise_l_bfgs_b(objective, vs, iters=iters, trace=True)
vs.print()

# Plot predictions.
mu, std = predict(build(vs))
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
plt.plot(t, mu + 3*std + vs['sigma'], c='tab:green', ls='--')
plt.plot(t, mu - 3*std - vs['sigma'], c='tab:green', ls='--')
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('$%.1f$'))
wbml.plot.tweak(legend=True)

# Plot kernel.
pred = predict_kernel(build(vs))
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
mean, lower, upper = predict_fourier(build(vs))
plt.subplot(2, 3, 5).set_title('Cosine Features')
freqs = model.ms/(model.b - model.a)
inds = np.concatenate(np.where(model.ms == 0) +
                      np.where(model.ms <= model.m_max))
plt.errorbar(freqs[inds], mean[inds], (mean[inds] - lower[inds],
                                       upper[inds] - mean[inds]),
             ls='none', marker='o', capsize=3)
plt.xlabel('Frequency (Hz)')
wbml.plot.tweak()

plt.subplot(2, 3, 6).set_title('Sine Features')
freqs = np.maximum(model.ms - model.m_max, 0)/(model.b - model.a)
inds = np.concatenate(np.where(model.ms == 0) +
                      np.where(model.ms > model.m_max))
plt.errorbar(freqs[inds], mean[inds], (mean[inds] - lower[inds],
                                       upper[inds] - mean[inds]),
             ls='none', marker='o', capsize=3)
plt.xlabel('Frequency (Hz)')
wbml.plot.tweak()

plt.tight_layout()
plt.show()
