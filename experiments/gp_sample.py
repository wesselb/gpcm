import lab.tensorflow as B
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wbml.out
import wbml.plot
from matplotlib.ticker import FormatStrFormatter
from stheno import GP, EQ, Delta
from varz import Vars
from varz.tensorflow import minimise_l_bfgs_b

from gpcm.elbo import construct, elbo, predict, predict_fourier, predict_kernel
from gpcm.gprv.model import GPRV
from gpcm.util import autocorr

# Sample data from a GP:
n = 200
t = B.linspace(0, 70, n)
y = B.flatten(GP(EQ() + 0.05*Delta())(t).sample())

alpha = 1/4  # Inverse length scale of window w.
lam = alpha/2  # Inverse length scale of input x.
alpha_t = (2*alpha)**.5  # Set prior kernel to variance one.

# Determine inducing points for h.
n_u_max = 50
dt = t[1] - t[0]
dt_u = dt
t_max_u = 2/alpha  # Twice window length.
n_u = int(t_max_u/dt_u)
n_u = min(n_u_max, n_u)
t_u = np.linspace(0, 2/alpha_t, n_u)
wbml.out.kv('n_u', n_u)

# Inter-domain transform for h:
gamma = 1/(2*dt_u)
gamma_t = (2*gamma)**.5

# Spacing between `a` and `min(t)` must be at least `max(t_u)`.
a, b = min(t) - max(t_u), max(t)

# Settings for `ms`:
f_max = 5
m_cap = 50
wbml.out.kv('f_max', f_max)

# Cap by the maximum frequency given.
m_max = int(np.ceil(f_max*(b - a)))
m_max = min(m_max, m_cap)
ms = np.arange(2*m_max + 1)
wbml.out.kv('m_max', m_max)
wbml.out.kv('f_max', m_max/(b - a))

# Initialise variables.
vs = Vars(tf.float64)
vs.bnd(alpha_t, name='alpha_t')
vs.bnd(alpha, name='alpha')
vs.bnd(lam, name='lambda')
vs.bnd(gamma_t, name='gamma_t')
vs.bnd(gamma, name='gamma')
vs.bnd(5e-2, lower=1e-3, upper=1e-1, name='sigma')
vs.get(shape=(n_u,), name='mu_u')
vs.pd(shape=(n_u, n_u), name='cov_u')


def build(vs):
    sigma = vs['sigma']
    alpha_t = vs['alpha_t']
    alpha = vs['alpha']
    gamma_t = vs['gamma_t']
    gamma = vs['gamma']
    lam = vs['lambda']
    mu_u = vs['mu_u'][:, None]
    cov_u = vs['cov_u']

    model = GPRV(lam=lam,
                 alpha=alpha,
                 alpha_t=alpha_t,
                 gamma=gamma,
                 gamma_t=gamma_t,
                 a=a,
                 b=b,
                 m_max=m_max,
                 ms=ms,
                 t_u=t_u)

    c = construct(model, t, y, sigma, mu_u, cov_u)

    return c


# Perform optimisation.
vs.print()
minimise_l_bfgs_b(tf.function(lambda vs_: -elbo(build(vs_)), autograph=False),
                  vs, iters=200, trace=True)
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
plt.plot(pred.t, EQ()(pred.t).mat[0, :], c='black', label='True')
plt.fill_between(pred.t, pred.err_68[0], pred.err_68[1],
                 facecolor='tab:green', alpha=0.15)
plt.fill_between(pred.t, pred.err_95[0], pred.err_95[1],
                 facecolor='tab:green', alpha=0.15)
plt.fill_between(pred.t, pred.err_99[0], pred.err_99[1],
                 facecolor='tab:green', alpha=0.15)
lags = int(max(pred.t/dt))
plt.plot(np.arange(lags + 1)*dt, autocorr(y, lags),
         c='tab:blue', label='Autocorrelation')
plt.plot(pred.t, pred.samples, c='tab:red', lw=1)
plt.scatter(t_u, 0*t_u, s=5, c='black')
wbml.plot.tweak(legend=True)

# Plot Fourier features.
mean, lower, upper = predict_fourier(build(vs))
plt.subplot(2, 3, 5).set_title('Cosine Features')
freqs = ms/(b - a)
inds = np.concatenate(np.where(ms == 0) + np.where(ms <= m_max))
plt.errorbar(freqs[inds], mean[inds], (mean[inds] - lower[inds],
                                       upper[inds] - mean[inds]),
             ls='none', marker='o', capsize=3)
plt.xlabel('Frequency (Hz)')
wbml.plot.tweak()

plt.subplot(2, 3, 6).set_title('Sine Features')
freqs = np.maximum(ms - m_max, 0)/(b - a)
inds = np.concatenate(np.where(ms == 0) + np.where(ms > m_max))
plt.errorbar(freqs[inds], mean[inds], (mean[inds] - lower[inds],
                                       upper[inds] - mean[inds]),
             ls='none', marker='o', capsize=3)
plt.xlabel('Frequency (Hz)')
wbml.plot.tweak()

plt.tight_layout()
plt.show()
