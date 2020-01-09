import lab as B

from .gprv.integrals import i_hx, i_ux
from .gprv.kernel_approx import K_u

__all__ = ['kernel_approx_u']


def kernel_approx_u(model, t1, t2, u):
    """Kernel approximation using inducing variables :math:`u` for the impulse
    response :math:`h`.

    Args:
        model (:class:`.model.GPRV`): Model.
        t1 (tensor): First time input.
        t2 (tensor): Second time input.
        u (tensor): Values of the inducing variables.

    Returns:
        tensor: Approximation of the kernel matrix broadcasted over `t1` and
            `t2`.
    """
    # Construct the first part.
    part1 = i_hx(model, t1[:, None], t2[None, :])

    # Construct the second part.
    Lu = B.cholesky(K_u(model))
    iLu = B.dense(B.trisolve(Lu, B.eye(Lu)))
    iLu_u = B.mm(iLu, u[:, None])
    Kiux = i_ux(model,
                # Increase ranks for broadcasting.
               t1[:, None, None, None],
               t2[None, :, None, None],
               model.t_u[None, None, :, None],
               model.t_u[None, None, None, :])
    trisolved = B.mm(B.mm(iLu, Kiux, tr_a=True), iLu)
    part2 = B.trace(trisolved, axis1=2, axis2=3) - \
            B.trace(B.mm(B.mm(iLu_u, trisolved, tr_a=True), iLu_u),
                    axis1=2, axis2=3)

    return part1 - part2

