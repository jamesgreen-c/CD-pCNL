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
        return -phi * x
    
    def diffusion(t, x):
        return sigma
    
    return drift, diffusion


@partial(jax.jit, static_argnums=(3))
def get_data(key: PRNGKey, phi: float, sigma: float, dim: int, dT: Array):
    """
    Produce continuous time LGSSM data where
        dX_t = -phi X_t dt + sigma dW_t
        Y_k | X_{t_k} ~ N(X_{t_k}, I)
    where dT[k] = t_{k+1} - t_k.

    Parameters
    ----------

    Returns
    -------
    xs: (T, dim) Latent states x_0, ..., x_{T-1}
    ys: (T, dim) Observations y_0, ..., y_{T-1}
    As: (T, 1) Persistence for each timestep
    Qs: (T, 1) STD for each timestep 
    """

    init_key, sampling_key = jax.random.split(key)
    T = dT.shape[0]  # number of timesteps rather than horizon time

    x0 = (sigma / jnp.sqrt(2 * phi)) * jax.random.normal(init_key, (dim,))
    eps_xs, eps_ys = jax.random.normal(sampling_key, (2, T, dim))

    # continuous time dynamics
    As = jnp.exp(-phi * dT)
    Qs = sigma * jnp.sqrt((1.0 - jnp.exp(-2.0 * phi * dT)) / (2.0 * phi))

    def body(x_k, inps):
        eps_x, eps_y, At, Qt = inps
        y_k = x_k + eps_y
        x_kp1 = At * x_k + Qt * eps_x
        return x_kp1, (x_k, y_k)
    
    _, (xs, ys) = jax.lax.scan(body, x0, (eps_xs, eps_ys, As, Qs))
    return xs, ys, As, Qs



@partial(jnp.vectorize, signature="(n),(n)->()", excluded=(2, 3, 4))
def log_potential(z, y, drift: Callable, diffusion: Callable, params):
    e_t_m_1, e_t, t, dt = params

    def _cov(t, x):
        sig = diffusion(t, x)
        return sig @ sig.T

    val = norm.logpdf(y, x)
    val += norm.logpdf(e_t, e_t_m_1, scale=diffusion(0, e_t)).sum()
    
    val += 0.5 * logdet(_cov(0, e_t_m_1)) - logdet(_cov(dt, e_t))
    val -= mvn_logpdf(
        e_t, 
        e_t_m_1 + drift(t, e_t_m_1) * dt,
        jnp.linalg.cholesky(dt * _cov(t, e_t_m_1))
    )

    x = bridge.euler(diffusion, z, e_t_m_1, e_t, dt)
    val += delyonhu(x, drift, diffusion, t, dt)

    return jnp.sum(val)


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
    from cd_ssm.utils.numerics import euler

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
