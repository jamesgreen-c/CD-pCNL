from typing import Callable
from functools import partial

from chex import Array, PRNGKey
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, lax
import jax


def drift(t, x, T, xT):
    return (xT - x) / (T - t)


@partial(jnp.vectorize, signature="(m,d),(d),(d)->(k,d)", excluded=(0, 4, 5))
def to_path(
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
    ws:         The driving Brownian motion. Shape (N, num, dw, ...)
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
    T = t0 + t
    ts = t0 + jnp.arange(num) * dt
    dws = ws[1:] - ws[:-1]

    def _body(x_k, inps):
        t_k, dw_k = inps
        x_kp1 = x_k + drift(t_k, x_k, T, xT) * dt + diffusion(t_k, x_k) * dw_k
        return x_kp1, x_kp1

    _, xs = lax.scan(_body, x0, (ts, dws))
    # xs = jnp.insert(xs, 0, x0, axis=0)
    return xs