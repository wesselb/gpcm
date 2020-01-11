import lab as B
import numpy as np
from matrix import Dense
from stheno import Normal

from gpcm.expq import EQ, const, var
from .model import Model
from .util import method

__all__ = ['GPCM']


def length_scale(ls):
    return (.5*np.pi)*(.5/ls**2)


class GPCM(Model):
    def __init__(self,
                 vs,
                 noise=1e-4,
                 alpha=None,
                 alpha_t=None,
                 window=None,
                 gamma=None,
                 scale=None,
                 omega=None,
                 n_u=None,
                 t_u=None,
                 n_z=None,
                 t_z=None,
                 t=None):
        Model.__init__(self)

        # First initialise optimisable model parameters.
        if alpha is None:
            alpha = 2*length_scale(window)

        if gamma is None:
            gamma = length_scale(scale) - 0.5*alpha

        if alpha_t is None:
            alpha_t = B.sqrt(2*alpha/B.pi)

        if omega is None:
            omega = length_scale(scale/2)

        self.noise = vs.positive(noise, name='noise')
        self.alpha = alpha  # Don't learn the window length.
        self.alpha_t = vs.positive(alpha_t, name='alpha_t')
        self.gamma = vs.positive(gamma, name='gamma')
        self.omega = vs.positive(omega, name='omega')

        self.vs = vs
        self.dtype = vs.dtype

        # Then initialise fixed variables.
        if t_u is None:
            t_u_max = 2/self.alpha
            t_u = B.linspace(0, t_u_max, n_u)

        if n_u is None:
            n_u = B.shape(t_u)[0]

        if t_z is None:
            t_z = B.linspace(min(t), max(t), n_z)

        if n_z is None:
            n_z = B.shape(t_z)[0]

        self.n_u = n_u
        self.t_u = t_u
        self.n_z = n_z
        self.t_z = t_z

        # Initialise variational parameters.
        mu_u = vs.unbounded(B.ones(self.n_u, 1), name='mu_u')
        cov_u = vs.positive_definite(B.eye(self.n_u), name='cov_u')
        self.q_u = Normal(cov_u, mu_u)

        # And finally initialise kernels.
        def k_h(t1, t2):
            return EQ(-const(self.alpha)*(t1**2 + t2**2) +
                      -const(self.gamma)*(t1 - t2)**2,
                      const=self.alpha_t)

        def k_xs(t1, t2):
            return EQ(-const(self.omega)*(t1 - t2)**2)

        self.k_h = k_h
        self.k_xs = k_xs


@method(GPCM)
def compute_K_u(model):
    return Dense(model
                 .k_h(var('t_u_1'), var('t_u_2'))
                 .eval(t_u_1=model.t_u[:, None],
                       t_u_2=model.t_u[None, :]))


@method(GPCM)
def compute_K_z(model):
    t_z_1 = model.t_z[:, None]
    t_z_2 = model.t_z[None, :]
    return (B.sqrt(0.5*B.pi/model.omega)*
            Dense(-0.5/model.omega*(t_z_1 - t_z_2)**2))


@method(GPCM)
def compute_i_hx(model, t1, t2):
    expq = model.k_h(var('t1') - var('tau'), var('t2') - var('tau'))
    return expq.integrate_box(('tau', -np.inf, np.inf),
                              t1=t1,
                              t2=t2)


@method(GPCM)
def compute_I_ux(model, t1, t2):
    expq = (model.k_h(var('t1') - var('tau'), var('t_u_1'))*
            model.k_h(var('t_u_2'), var('t2') - var('tau')))
    return expq.integrate_box(('tau', -np.inf, np.inf),
                              t1=t1[:, None, None, None],
                              t2=t2[None, :, None, None],
                              t_u_1=model.t_u[None, None, :, None],
                              t_u_2=model.t_u[None, None, None, :])


@method(GPCM)
def compute_I_hz(model, t1, t2):
    expq = (model.k_h(var('t1') - var('tau1'), var('t2') - var('tau2'))*
            model.k_xs(var('tau1'), var('t_z_1'))*
            model.k_xs(var('t_z_2'), var('tau2')))
    return expq.integrate_box(('tau1', -np.inf, np.inf),
                              ('tau2', -np.inf, np.inf),
                              t1=t1[:, None, None, None],
                              t2=t2[None, :, None, None],
                              t_z_1=model.t_z[None, None, :, None],
                              t_z_2=model.t_z[None, None, None, :])


@method(GPCM)
def compute_I_uz(model, t):
    expq = (model.k_h(var('t') - var('tau'), var('t_u'))*
            model.k_xs(var('tau'), var('t_z')))
    return expq.integrate_box(('tau', -np.inf, np.inf),
                              t=t[:, None, None],
                              t_u=model.t_u[None, :, None],
                              t_z=model.t_z[None, None, :])
