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
from cd_ssm import brownian


def get_dynamics(phi: float, sigma: float):

    def drift(t, x):
        x = jnp.atleast_1d(x)
        return -phi * x
    
    def diffusion(t, x):
        return sigma
    
    return drift, diffusion


@partial(jax.jit, static_argnums=(3, 5))
def get_data(
        key: PRNGKey, 
        phi: float, 
        sigma: float, 
        dim: int, 
        dts: Array,
        num: int
    ):
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
    num:   The number of discretisation steps for each path at time t
    
    Returns
    -------
    xs: (K, dim) Latent states x_0, ..., x_{K-1}
    ys: (K, dim) Observations y_0, ..., y_{K-1}
    As: (K, 1) Persistence for each timestep
    Qs: (K, 1) STD for each timestep 
    """

    init_key, wiener_key, sampling_key = jax.random.split(key, 3)
    K = dts.shape[0]  # number of timesteps rather than horizon time

    # needed for exact smoothing on endpoints if useful
    chol_P0 = (sigma / jnp.sqrt(2 * phi))
    chol_Qs = sigma * jnp.sqrt((1.0 - jnp.exp(-2.0 * phi * dts)) / (2.0 * phi))
    As = jnp.exp(-phi * dts)

    ep_0 = jnp.zeros((dim,))
    e_0 = chol_P0 * jax.random.normal(init_key, (dim,))
    
    w_keys = jax.random.split(wiener_key, K)
    us = jax.vmap(lambda _k, _dt: brownian.simulate(_k, jnp.zeros((dim, )), _dt, num, 1))(w_keys, dts)

    eps_es, eps_ys = jax.random.normal(sampling_key, (2, K, dim))
    def body(e_k, inps):
        eps_e, eps_y, At, Qt = inps
        y_k = e_k + eps_y
        e_kp1 = At * e_k + Qt * eps_e
        return e_kp1, (e_k, y_k)
    
    _, (es, ys) = jax.lax.scan(body, e_0, (eps_es, eps_ys, As, chol_Qs))
    xs = (us, es)
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
    num = 50
    xs, ys, *_ = get_data(key, phi, sigma, dim, dT, num)

    # convert to path-space
    us, es = xs
    us = jnp.swapaxes(us, -1, -2)
    ep_0 = jnp.zeros((dim,))
    es = jnp.insert(es, 0, ep_0, axis=0)
    ts = jnp.cumsum(dT)

    _drift, _diffusion = get_dynamics(phi, sigma)
    paths = jax.vmap(bridge.to_path, in_axes=(None, 0, 0, 0, 0, 0))(_diffusion, us, es[:-1], es[1:], ts, dT)

    # plot path
    K, num, D = paths.shape
    es = paths[:, -1, ...]
    paths = paths.reshape((K*num, D))

    x_axis = jnp.arange(K*num)
    obs_ts = K * num * ts / 100
    # obs_ts = jnp.insert(jnp.zeros((dim, )), 0, obs_ts, axis=0)

    print(es.shape, obs_ts.shape, ys.shape)
    
    plt.figure(figsize=(15, 5))
    plt.plot(x_axis, paths, color="black", label="latents")
    plt.scatter(obs_ts, es, color="blue", label="endpoints")
    plt.scatter(obs_ts, ys, color="red", marker="x", label="observations")
    plt.legend()
    plt.savefig("data.png")
    plt.close()
