from functools import partial

from chex import Array, PRNGKey

import jax.numpy as jnp
from jax.scipy.stats import norm

from cd_ssm import brownian
from cd_ssm.utils.cov import wiener_covariance


def propose(
        key: PRNGKey, 
        xp: Array, 
        rho: Array,
        grad: Array,
        dt: Array, 
        N: Array
    ):
    """
    Propose N new paths according to:
         x = rho*xp + (1 - rho) * C @ grad + sqrt(1 - rho^2) * W(R^D)
    where W is a sample from the Weiner law on R^D

    Parameters
    ----------
    key: RNG
    xp:   Array  (mesh, D)
    rho:  Array
    grad: Array  (mesh, D)
    dt:   Array  The time delta
    N:    Array

    Returns
    -------
    x:   Array (N, mesh, D)
    """
    mesh, D = xp.shape[-2:]
    
    C = wiener_covariance(dt, mesh)
    C_grad = C @ grad
    
    w0 = jnp.zeros((N, D))
    w = brownian.simulate(key, w0, dt, mesh - 1, N)
    
    return rho * xp + (1 - rho) * C_grad + jnp.sqrt(1 - jnp.square(rho)) * w


@partial(jnp.vectorize, signature="(m,d),(m,d),(),(m,d),()->()")
def logpdf(
        xp: Array,
        x: Array,
        rho: Array,
        grad: Array,
        dt: Array,
    ):
    """
    Log-density of the discretised pCN proposal on a Brownian path.

    The proposal is
        x = rho * xp + (1 - rho) * C @ grad + sqrt(1 - rho^2) * w
    where w is a Brownian path started at 0.

    Hence, on the discretisation grid:
        x[0] = rho * xp[0] + (1-rho) * C @ ∇Gamma(x)    (deterministic)
        dx_k ~ N(rho * dxp_k, (1-rho^2)*subdt*I),   k=1,...,mesh-1

    Parameters
    ----------
    xp:   Array  (mesh, D)
    x:    Array  (mesh, D)
    rho:  Array
    grad: Array  (mesh, D)
    dt:   Array  The time delta
    """
    dx  = x[1:]  - x[:-1]    # (mesh-1, D)

    mesh, D = xp.shape[-2:]
    subdt = dt / (mesh - 1)

    C = wiener_covariance(dt, mesh)
    C_grad = C @ grad

    scale = jnp.sqrt((1.0 - rho**2) * subdt)
    mean = rho * xp + (1.0 - rho) * C_grad
    dmean = mean[1:] - mean[:-1]
    return norm.logpdf(dx, dmean, scale).sum()
    