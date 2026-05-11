from typing import Callable
from functools import partial

from chex import Array, PRNGKey
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, lax
from jax.scipy.stats import norm



def simulate(
        key: PRNGKey, 
        w0: Array, 
        t: Array,
        num: Array,
        N: Array
    ):
    """
    Implementation of an exact simulate for standard Brownian motion
    
    Parameters
    ----------
    key:        RNG
    w0:         The starting point. Shape (N', d, ...) for N particles. 
    t:          The amount of time that passes
    num:        The number of steps to take in dt time
    N:          The number of particles
    
    Returns
    ----------
    ws:         The resulting Brownian motion (N, d, ..., num)
    """

    eps = jr.normal(key, (num, N, *w0.shape[1:]))
    dt = t / num

    def _body(w_k, eps_k):
        w_k_p_1 = w_k + jnp.sqrt(dt) * eps_k
        return w_k_p_1, w_k_p_1

    _, ws = lax.scan(_body, w0, eps)
    ws = jnp.insert(ws, 0, w0, axis=0)
    ws = jnp.swapaxes(ws, 0, 1)
    return ws


def propose(
        key: PRNGKey, 
        xp: Array, 
        delta: Array, 
        dt: Array, 
        N: Array
    ):
    """
    Propose N new paths according to:
         x = xp + sqrt(delta) * W(R^D)
    where W is a sample from the Weiner law on R^D

    Parameters
    ----------
    key:   RNG
    xp:    Array  (mesh, D)
    delta: Array
    dt:    Array  The time delta
    N:     Array

    Returns
    -------
    x:   Array (N, mesh, D)
    """
    mesh, D = xp.shape[-2:]

    w0 = jnp.zeros((N, D))
    w = simulate(key, w0, dt, mesh - 1, N)
    return xp + jnp.sqrt(delta) * w


@partial(jnp.vectorize, signature="(m,d),(m,d),(),()->()")
def logpdf(
        xp: Array,
        x: Array,
        delta: Array,
        dt: Array,
    ):
    """
    Log-density of the discretised Brownian random-walk proposal.

    The proposal is
        x = xp + sqrt(delta) * W,
    where W is a Brownian path started at 0.

    Therefore, on the discretisation grid:
        x[0] = xp[0],
        dx_k = dxp_k + sqrt(delta) * dW_k,
        dx_k | dxp_k ~ N(dxp_k, delta * subdt * I),
        k = 1, ..., mesh - 1.

    Parameters
    ----------
    xp:     Array  (mesh, D). Previous/reference path.
    x:      Array  (mesh, D). Proposed path.
    delta:  Array. Random-walk variance scale.
    dt:     Array. Time interval length.

    Returns
    -------
    val:    Array. Log-density of x given xp, up to the deterministic initial point.
    """
    dxp = xp[1:] - xp[:-1]
    dx = x[1:] - x[:-1]

    mesh, _ = xp.shape[-2:]
    subdt = dt / (mesh - 1)
    scale = jnp.sqrt(delta * subdt)

    return norm.logpdf(dx, loc=dxp, scale=scale).sum()