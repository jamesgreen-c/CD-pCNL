"""
Continuous time linear Gaussian model
"""

from functools import partial
from typing import Callable

import jax
from jax import numpy as jnp
from jax.scipy.stats import norm
from chex import PRNGKey, Array

from cd_ssm.utils.math import logdet, mvn_logpdf
from cd_ssm.delyon_hu import delyonhu
from cd_ssm import bridge


def get_dynamics(phi: float, sigma: float):

    def drift(t, x):
        x = jnp.atleast_1d(x)
        return -phi * x
    
    def diffusion(t, x):
        return sigma
    
    return drift, diffusion


@partial(jax.jit, static_argnums=(3))
def get_data(key: PRNGKey, phi: float, sigma: float, dim: int, dts: Array):
    """
    Produce continuous time LGSSM data where
        dX_t = -phi X_t dt + sigma dW_t
        Y_k | X_{t_k} ~ N(X_{t_k}, I)
    where dts[k] = t_{k+1} - t_k.

    Parameters
    ----------
    key:   PRNGKey
    phi:   Persistence parameter
    sigma: Standard deviation of prior dynamics
    dim:   Dimension of latent state
    dts:   (K,)  The time deltas for all K steps

    Returns
    -------
    xs: (K, dim) Latent states x_0, ..., x_{K-1}
    ys: (K, dim) Observations y_0, ..., y_{K-1}
    As: (K, 1) Persistence for each timestep
    Qs: (K, 1) STD for each timestep 
    """

    init_key, sampling_key = jax.random.split(key)
    K = dts.shape[0]  # number of timesteps rather than horizon time

    chol_P0 = (sigma / jnp.sqrt(2 * phi))
    x0 = chol_P0 * jax.random.normal(init_key, (dim,))
    eps_xs, eps_ys = jax.random.normal(sampling_key, (2, K, dim))

    # continuous time dynamics
    As = jnp.exp(-phi * dts)
    chol_Qs = sigma * jnp.sqrt((1.0 - jnp.exp(-2.0 * phi * dts)) / (2.0 * phi))

    def body(x_k, inps):
        eps_x, eps_y, At, Qt = inps
        y_k = x_k + eps_y
        x_kp1 = At * x_k + Qt * eps_x
        return x_kp1, (x_k, y_k)
    
    _, (xs, ys) = jax.lax.scan(body, x0, (eps_xs, eps_ys, As, chol_Qs))
    return xs, ys, As, chol_Qs, chol_P0





@partial(jnp.vectorize, signature="(m,d),(d)->()", excluded=(2, 3, 4, 5, 6))
def log_potential(x, ep, y, drift: Callable, diffusion: Callable, t: Array, dt: Array):
    e = x[-1]

    def _cov(t, x):
        sig = diffusion(t, x) * jnp.eye(x.shape[0])
        return sig @ sig.T
    
    val = norm.logpdf(y, e).sum()
    val += norm.logpdf(e, ep, scale=diffusion(0, ep)).sum()
    val += 0.5 * (logdet(_cov(0, ep)) - logdet(_cov(dt, e)))
    val += delyonhu(x, drift, diffusion, t, dt)
    return val


# def log_likelihood(x, y):
#     return jnp.sum(log_potential(x, y))


# def log_pdf(xs, ys, sigma):
#     def _logpdf(zs):
#         out = jnp.sum(norm.logpdf(zs[0], scale=sigma))
#         out += jnp.sum(norm.logpdf(zs[1:], zs[:-1], sigma))
#         out += jnp.sum(norm.logpdf(zs, ys))
#         return out

#     return jnp.vectorize(_logpdf, signature="(T,d)->()")(xs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from cd_ssm.euler import euler

    key = jax.random.PRNGKey(0)
    phi = 0.5
    sigma = 1
    dim = 1
    T = 1
    dT = jnp.repeat(1/T, 100)
    xs, ys, *_ = get_data(key, phi, sigma, dim, dT)

    # compare to euler scheme
    _drift, _diffusion = get_dynamics(phi, sigma)
    xs_euler = euler(key, _drift, _diffusion, xs[0], 0, 100, 99, 1)
    print(xs.shape, xs_euler.shape)

    ts = jnp.cumsum(dT)
    plt.figure(figsize=(15, 5))
    plt.plot(ts, xs, color="black", linestyle="--", label="latents")
    plt.plot(ts, xs_euler, color="blue", alpha=0.5, label="euler")
    plt.scatter(ts, ys, color="red", marker="x", label="observations")
    plt.legend()
    plt.savefig("data.png")
    plt.close()
