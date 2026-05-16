from functools import partial

from chex import Array, PRNGKey

import jax.numpy as jnp
from jax.scipy.stats import norm

from cd_ssm import brownian


def propose(
        key: PRNGKey, 
        xp: Array, 
        rho: Array, 
        dt: Array, 
        N: Array
    ):
    """
    Propose N new paths according to:
         x = rho*xp + sqrt(1 - rho^2) * W(R^D)
    where W is a sample from the Weiner law on R^D

    Parameters
    ----------
    key: RNG
    xp:  Array  (mesh, D)
    rho: Array
    dt:  Array  The time delta
    N:   Array

    Returns
    -------
    x:   Array (N, mesh, D)
    """
    mesh, D = xp.shape[-2:]

    w0 = jnp.zeros((N, D))
    w = brownian.simulate(key, w0, dt, mesh - 1, N)
    return rho * xp + jnp.sqrt(1 - jnp.square(rho)) * w


@partial(jnp.vectorize, signature="(m,d),(m,d),(),()->()")
def logpdf(
        xp: Array,
        x: Array,
        rho: Array,
        dt: Array,
    ):
    """
    Log-density of the discretised pCN proposal on a Brownian path.

    The proposal is
        x = rho * xp + sqrt(1 - rho^2) * w
    where w is a Brownian path started at 0.

    Hence, on the discretisation grid:
        x[0] = rho * xp[0]                      (deterministic)
        dx_k ~ N(rho * dxp_k, (1-rho^2)*subdt*I),   k=1,...,mesh-1

    Parameters
    ----------
    xp:  Array  (mesh, D)
    x:   Array  (mesh, D)
    rho: Array
    dt:  Array  The time delta
    """
    dxp = xp[1:] - xp[:-1]   # (mesh-1, D)
    dx  = x[1:]  - x[:-1]    # (mesh-1, D)

    mesh, D = xp.shape[-2:]
    subdt = dt / (mesh - 1)
    scale = jnp.sqrt((1.0 - rho**2) * subdt)

    return norm.logpdf(dx, rho * dxp, scale).sum()

    