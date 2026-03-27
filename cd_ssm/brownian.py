from typing import Callable

from chex import Array, PRNGKey
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, lax


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
