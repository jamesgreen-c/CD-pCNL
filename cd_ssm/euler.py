from typing import Callable
from functools import partial

from chex import Array, PRNGKey
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, lax

from cd_ssm.utils.math import mvn_logpdf


def euler(
        key: PRNGKey,
        drift: Callable, 
        diffusion: Callable, 
        x0: Array,
        t0: Array, 
        t: Array, 
        num: Array, 
        N: Array
    ):
    """
    Implementation of the euler-maruyama scheme for approximating solutions to SDEs
    
    Parameters
    ----------
    key:        RNG
    drift:      The drift function. Should take (t, x) as args
    diffusion:  The diffusion function. Should take (t, x) as args
    x0:         The starting point. Shape (N, d, ...) for N particles.
    t0:         The starting time
    t:          The amount of time that passes
    num:        The number of steps to take in dt time
    N:          The number of particles
    
    Returns
    ----------
    xs:         The resulting interpolated paths (N, d, ..., num)
    """

    eps = jr.normal(key, (num, N, *x0.shape[1:]))
    dt = t / num
    ts = t0 + jnp.arange(num) * dt

    def _body(x_k, inps):
        t_k, eps_k = inps
        x_k_p_1 = x_k + drift(t_k, x_k)*dt + diffusion(t_k, x_k) * jnp.sqrt(dt) * eps_k
        return x_k_p_1, x_k_p_1
    
    _, xs = lax.scan(_body, x0, (ts, eps))
    xs = jnp.insert(xs, 0, x0, axis=0)
    xs = jnp.swapaxes(xs, 0, 1)
    return xs


@partial(jnp.vectorize, signature="(d),(d),(),()->()", excluded=(2, 3))
def logpdf(x, xp, drift: Callable, diffusion: Callable, t: Array, dt: Array):
    
    def _cov(t, x):
        x = jnp.atleast_1d(x)
        sig = diffusion(t, x) * jnp.eye(x.shape[0])
        return sig @ sig.T
    
    return mvn_logpdf(
        x, 
        xp + drift(t, xp) * dt,
        jnp.linalg.cholesky(dt * _cov(t, xp))
    )
