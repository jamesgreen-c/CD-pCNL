from typing import Callable

from chex import Array, PRNGKey
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, lax


def drift(t, x, dt, xT):
    return (xT - x) / (dt - t)


def euler(
        diffusion: Callable,
        ws: Array,
        x0: Array,
        xT: Array,
        t0: Array,
        t: Array,
    ):
    """
    Implementation of the euler-maruyama scheme for approximating solutions to SDEs when the driving noise is given
    
    Parameters
    ----------
    diffusion:  The diffusion function. Should take (t, x) as args
    ws:         The driving Brownian motion. Shape (num, N, dw, ...)
    x0:         The starting point. Shape (N, d, ...) for N particles.
    xT:         The end point. Shape (N, d, ...) for N particles.
    t0:         The starting time
    t:          The amount of time that passes
    num:        The number of steps to take in dt time
    N:          The number of particles
    
    Returns
    ----------
    xs:         The resulting interpolated paths (N, num, d, ...)
    """
    num = ws.shape[0] - 1
    dt = t / num
    ts = t0 + jnp.arange(num + 1) * dt

    def _body(x_k, inps):
        t_k, w_k = inps
        x_k_p_1 = x_k + drift(t_k, x_k, dt, xT)*dt + diffusion(t_k, x_k) * jnp.sqrt(dt) * w_k
        return x_k_p_1, x_k_p_1
    
    _, xs = lax.scan(_body, x0, (ts, ws))
    xs = jnp.insert(xs, 0, x0, axis=0)
    xs = jnp.swapaxes(xs, 0, 1)
    return xs