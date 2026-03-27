from enum import Enum
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.stats import norm


from experiments.lgssm.model import log_potential

from cd_ssm.utils.math import mvn_logpdf
from cd_ssm.utils.mcmc_utils import aux_sampling_routine
from cd_ssm import euler
from cd_ssm import brownian as br
from cd_ssm import t_csmc
from cd_ssm import bridge


class KernelType(Enum):
    CSMC = 0
    PCN = 1

    @property
    def kernel_maker(self):
        if self == KernelType.CSMC:
            return get_csmc_kernel
        elif self == KernelType.PCN:
            return get_pcn_csmc_kernel
        else:
            raise NotImplementedError

    def shape_delta(self, delta, T):
        if self == KernelType.CSMC:
            return delta
        elif self == KernelType.PCN:
            return delta * np.ones((T,))
        else:
            return NotImplementedError("Shape delta not implemented for kernel type")
        
#######################
# Kernel constructors #
#######################


def get_csmc_kernel(ys, drift: Callable, diffusion: Callable, sigma, N, num, dts, style="guided", **kwargs):
    """
    Implementation of a conditional sequential Monte Carlo kernel for continuous-discrete state-space models.
    Uses guided bridge proposals for the forward particle system and then applies a backward pass to sample
    a full ancestral trajectory.

    Parameters
    ----------
    ys:         The observations at the discrete times. Shape (T, d)
    drift:      The drift function. Should take (t, x) as args
    diffusion:  The diffusion function. Should take (t, x) as args
    sigma:      The standard deviation of the initial Gaussian prior
    N:          The number of particles, excluding the retained reference path
    num:        The number of mesh steps used within each observation interval
    dts:        The time increments between observations. Shape (T,)
    style:      The proposal style to use. Currently only "guided" is implemented
    kwargs:     Additional keyword arguments passed to the underlying cSMC kernel,
                such as resampling and ancestor move functions

    Returns
    ----------
    kernel:     The cSMC kernel. Takes a PRNG key and a state (reference path, reference ancestors),
                runs the forward pass and backward pass, and returns the updated particle genealogy
    init:       Initialiser for the cSMC state. Takes a reference path and returns the pair
                (reference path, zero ancestor indices)
    """
    T, dx = ys.shape
    ts = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dts)])[:-1]

    if style == "guided":

        def M0_rvs(key, _):
            key1, key2 = jr.split(key)
            u_00 = jnp.zeros((N+1, dx))
            u_0 = br.simulate(key1, u_00, dts[0], num, N + 1) 
            e_0 = sigma * jr.normal(key2, (N + 1, dx))
            return (u_0, e_0)

        def Mt_rvs(key, z_t_m_1, params):
            _, t, dt = params
            _, e_t_m_1 = z_t_m_1

            key1, key2 = jr.split(key)
            u_t0 = jnp.zeros((N + 1, dx))
            u_t = br.simulate(key1, u_t0, dt, num, N + 1)
            e_ts = euler.euler(key2, drift, diffusion, e_t_m_1, t, dt, 1, N + 1)
            return (u_t, e_ts[:, -1])

        M0_logpdf = lambda z: norm.logpdf(z[1], loc=0.0, scale=sigma).sum(axis=-1)
        Mt_logpdf = lambda z_t_m_1, z_t, params: jax.vmap(lambda ep, e: euler.logpdf(e, ep, drift, diffusion, params[1], params[2]))(z_t_m_1[1], z_t[1])

        def Gamma_0(z):
            _, e = z
            return norm.logpdf(ys[0], loc=e).sum(axis=-1) + M0_logpdf(z)

        def Gamma_t(z_t_m_1, z_t, params):
            y_t, t, dt = params
            _, e_t_m_1 = z_t_m_1
            u_t, e_t = z_t
            x_t = bridge.to_path(diffusion, u_t, e_t_m_1, e_t, t, dt)
            return log_potential(x_t, e_t_m_1, y_t, drift, diffusion, t, dt)
        
    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'guided'")

    M0 = M0_rvs, M0_logpdf
    Mt = Mt_rvs, Mt_logpdf, (ys[1:], ts[1:], dts[1:])
    Gamma_t_plus_params = Gamma_t, (ys[1:], ts[1:], dts[1:])

    kernel = lambda key, state, *_: t_csmc.kernel(key, state[0], state[1], M0, Gamma_0, Mt, Gamma_t_plus_params, N=N,
                                                **kwargs)
    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
        return aux_sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)

    return kernel, init, sampling_routine_fn


def get_filter_csmc_kernel(ys, drift: Callable, diffusion: Callable, sigma, N, num, dts, style="guided", **kwargs):
    """
    Implementation of a forward-only conditional sequential Monte Carlo kernel for continuous-discrete
    state-space models. Uses guided bridge proposals to generate weighted particles approximating the
    filtering distributions, but does not perform a backward pass.

    Parameters
    ----------
    ys:         The observations at the discrete times. Shape (T, d)
    drift:      The drift function. Should take (t, x) as args
    diffusion:  The diffusion function. Should take (t, x) as args
    sigma:      The standard deviation of the initial Gaussian prior
    N:          The number of particles, excluding the retained reference path
    num:        The number of mesh steps used within each observation interval
    dts:        The time increments between observations. Shape (T,)
    style:      The proposal style to use. Currently only "guided" is implemented
    kwargs:     Additional keyword arguments passed to the underlying forward pass,
                such as the resampling function and whether the run is conditional

    Returns
    ----------
    kernel:     The forward-only cSMC kernel. Takes a PRNG key and a state (reference path, reference ancestors),
                runs only the forward pass, and returns the weighted particle system and genealogy
    init:       Initialiser for the forward-pass state. Takes a reference path and returns the pair
                (reference path, zero ancestor indices)
    """
    T, dx = ys.shape
    ts = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dts)])[:-1]

    if style == "guided":

        def M0_rvs(key, _):
            key1, key2 = jr.split(key)
            u_00 = jnp.zeros((N+1, dx))
            u_0 = br.simulate(key1, u_00, dts[0], num, N + 1) 
            e_0 = sigma * jr.normal(key2, (N + 1, dx))
            return (u_0, e_0)

        def Mt_rvs(key, z_t_m_1, params):
            _, t, dt = params
            _, e_t_m_1 = z_t_m_1

            key1, key2 = jr.split(key)
            u_t0 = jnp.zeros((N + 1, dx))
            u_t = br.simulate(key1, u_t0, dt, num, N + 1)
            e_ts = euler.euler(key2, drift, diffusion, e_t_m_1, t, dt, 1, N + 1)
            return (u_t, e_ts[:, -1])

        M0_logpdf = lambda z: norm.logpdf(z[1], loc=0.0, scale=sigma).sum(axis=-1)
        Mt_logpdf = lambda z_t_m_1, z_t, params: jax.vmap(lambda ep, e: euler.logpdf(e, ep, drift, diffusion, params[1], params[2]))(z_t_m_1[1], z_t[1])

        def Gamma_0(z):
            _, e = z
            return norm.logpdf(ys[0], loc=e).sum(axis=-1) + M0_logpdf(z)

        def Gamma_t(z_t_m_1, z_t, params):
            y_t, t, dt = params
            _, e_t_m_1 = z_t_m_1
            u_t, e_t = z_t
            x_t = bridge.to_path(diffusion, u_t, e_t_m_1, e_t, t, dt)
            return log_potential(x_t, e_t_m_1, y_t, drift, diffusion, t, dt)
        
    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'guided'")

    M0 = M0_rvs, M0_logpdf
    Mt = Mt_rvs, Mt_logpdf, (ys[1:], ts[1:], dts[1:])
    Gamma_t_plus_params = Gamma_t, (ys[1:], ts[1:], dts[1:])

    kernel = lambda key, state, *_: t_csmc.forward_pass(key, state[0], state[1], M0, Gamma_0, Mt, Gamma_t_plus_params, N=N,
                                                **kwargs)
    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
        return aux_sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)

    return kernel, init, sampling_routine_fn



def get_pcn_csmc_kernel(ys, sigma, N, style="filtering", stop_gradient=False, **kwargs):
    pass

