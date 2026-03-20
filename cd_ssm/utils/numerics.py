from typing import Callable

from chex import Array, PRNGKey
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, lax


def euler(
        key: PRNGKey,
        drift: Callable, 
        diffusion: Callable, 
        x0: Array, 
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
    x0:         The starting point. Shape (N', d, ...) for N particles. (strictly speaking N' can be = 1 when N isn't)
    t:          The amount of time that passes
    num:        The number of steps to take in dt time
    N:          The number of particles
    
    Returns
    ----------
    xs:         The resulting interpolated paths (N, d, ..., num)
    """

    eps = jr.normal(key, (num, N, *x0.shape[1:]))
    dt = t / num
    ts = jnp.arange(num) * dt

    def _body(x_k, inps):
        t_k, eps_k = inps
        x_k_p_1 = x_k + drift(t_k, x_k)*dt + diffusion(t_k, x_k) * jnp.sqrt(dt) * eps_k
        return x_k_p_1, x_k_p_1
    
    _, xs = lax.scan(_body, x0, (ts, eps))
    xs = jnp.insert(xs, 0, x0)
    return xs
    

def integrate_dt(a: Array, b: Array, f: Callable, xs: Array):
    """
    Implementation of a left-point Riemann approximation scheme for time integrals
    involving functions of discretised SDE paths.

    Parameters
    ----------
    b:          The integral endpoint
    a:          The integral startpoint
    f:          The function to integrate. Must take (t, x) as args
    xs:         The discretised path, including both endpoints, shape (num, dx)

    Returns
    ----------
    Approximation to the time integral
    """
    num = xs.shape[0] - 1
    dt = (b - a) / num
    ts = a + jnp.arange(num) * dt
    fs = vmap(f)(ts, xs[:-1])
    return jnp.sum(fs * dt)


def integrate_dx(a: Array, b: Array, f: Callable, xs: Array):
    """
    Implementation of a left-point Riemann-Stieltjes approximation scheme for path integrals
    involving functions of discretised SDE paths.

    Parameters
    ----------
    b:          The integral endpoint
    a:          The integral startpoint
    f:          The function to integrate. Must take (t, x) as args
    xs:         The discretised path, including both endpoints, shape (num, dx)

    Returns
    ----------
    Approximation to the path integral
    """
    num = xs.shape[0] - 1
    dt = (b - a) / num
    ts = a + jnp.arange(num) * dt
    dx = xs[1:] - xs[:-1]
    fs = vmap(f)(ts, xs[:-1])
    return jnp.einsum("ni,ni->", fs, dx)


def integrate_dcov_inv(a: Array, b: Array, f: Callable, cov: Callable, xs: Array):
    """
    Implementation of a left-point Riemann-Stieltjes approximation scheme for covariance integrals
    involving functions of discretised SDE paths.

    Parameters
    ----------
    b:          The integral endpoint
    a:          The integral startpoint
    f:          The function to integrate. Must take (t, x) as args
    cov:        The covariance function. Must take (t, x) as args
    xs:         The discretised path, including both endpoints, shape (num, dx)

    Returns
    ----------
    Approximation to the path integral
    """
    num = xs.shape[0] - 1
    dt = (b - a) / num
    ts = a + jnp.arange(num + 1) * dt
    
    cov_invs = vmap(lambda t, x: jnp.linalg.inv(cov(t, x)))(ts, xs)
    dcov_invs = cov_invs[1:] - cov_invs[:-1]
    
    fs = vmap(f)(ts[:-1], xs[:-1])
    return jnp.sum(jnp.einsum("ni,nij,nj->", fs, dcov_invs, fs), axis=0)