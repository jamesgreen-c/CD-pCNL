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
from cd_ssm import euler
from cd_ssm import brownian as br
from cd_ssm import csmc
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

    """
    T, dx = ys.shape
    ts = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dts)])
    
    if style == "guided":

        def M0_rvs(key, _):
            """ Returns t0 distribution for pathspace not z-space """
            key1, key2, key3 = jr.split(key, 3)
            
            z_0 = br.simulate(key1, jnp.zeros((dx,)), dts[0], num-1, N+1)
            
            e0 = sigma * jr.normal(key2, (N + 1, dx))
            e_1s = euler.euler(key3, drift, diffusion, e0, ts[0], dts[0], 1, N+1)
            x = bridge.euler(diffusion, z_0, e0, e_1s[-1], 0, dts[0])
            return x
                
        def Mt_rvs(key, x_t_m_1, params):
            y_t, t, dt = params

            key1, key2 = jr.split(key)
            z_t = br.simulate(key1, jnp.zeros((dx,)), dt, num, N + 1)  # exact brownian simulation
            
            e_ts = euler.euler(key2, drift, diffusion, x_t_m_1[-1], t, dt, 1, N + 1)
            x = bridge.euler(diffusion, z_t, x_t_m_1[-1], e_ts[-1], t, dt)            
            return x[1:]  # remove duplicate endpoint

        M0_logpdf = lambda x: norm.logpdf(x[0], scale=sigma).sum() + euler.logpdf(x[-1], x[0], drift, diffusion)
        Mt_logpdf = lambda x_t_m_1, x_t, params: euler.logpdf(x_t, x_t_m_1, drift, diffusion, params[0], params[1])
        
        def Gamma_0(x):
            """ Returns discrete weight for x0 """
            sig = diffusion(0, x)
            cov = sig @ sig.T
            chol_Q = jnp.linalg.cholesky(cov)
            return mvn_logpdf(ys[0], x, chol_Q, constant=False) + M0_logpdf(x)
        
        def Gamma_t(x_t_m_1, x_t, params):
            y_t, t, dt = params
            return log_potential(x_t, x_t_m_1, y_t, drift, diffusion, t, dt)

    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'guided'")

    M0 = M0_rvs, M0_logpdf
    Mt = Mt_rvs, Mt_logpdf, (ys[1:], ts[1:], dts)
    Gamma_t_plus_params = Gamma_t, (ys[1:], ts[1:], dts)

    kernel = lambda key, state, *_: csmc.kernel(key, state[0], state[1], M0, Gamma_0, Mt, Gamma_t_plus_params, N=N,
                                                **kwargs)
    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    return kernel, init


def get_filter_csmc_kernel(ys, drift: Callable, diffusion: Callable, sigma, N, num, dts, style="guided", **kwargs):
    """

    """
    T, dx = ys.shape
    ts = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dts)])[:-1]

    if style == "guided":

        # def M0_rvs(key, _):
        #     """ Returns t0 distribution for pathspace not z-space """
        #     x0 = sigma * jr.normal(key, (N + 1, num + 1, dx))
        #     return x0
        
        def M0_rvs(key, _):
            e0 = sigma * jr.normal(key, (N + 1, dx))
            return jnp.repeat(e0[:, None, :], num + 1, axis=1)

        def Mt_rvs(key, x_t_m_1, params):
            y_t, t, dt = params

            key1, key2 = jr.split(key)
            z_t0 = jnp.zeros((N + 1, dx))
            z_t = br.simulate(key1, z_t0, dt, num, N + 1)  # exact brownian simulation
            
            e_ts = euler.euler(key2, drift, diffusion, x_t_m_1[:, -1], t, dt, 1, N + 1)
            x = bridge.euler(diffusion, z_t, x_t_m_1[:, -1], e_ts[:, -1], t, dt)            
            return x[:, 1:]  # remove duplicate endpoint

        
        M0_logpdf = lambda x: norm.logpdf(x[-1], scale=sigma).sum()
        M0_logpdf = jax.vmap(M0_logpdf)
        
        Mt_logpdf = lambda x_t_m_1, x_t, params: euler.logpdf(x_t[-1], x_t_m_1[-1], drift, diffusion, params[1], params[2])
        Mt_logpdf = jax.vmap(Mt_logpdf, in_axes=(0, 0, None))

        @jax.vmap
        def Gamma_0(x):
            """ Returns discrete weight for x0 """
            e = x[-1]
            obs_ll = jnp.sum(norm.logpdf(ys[0], loc=e))
            prior_ll = jnp.sum(norm.logpdf(e, scale=sigma))
            return obs_ll + prior_ll
            # cov = sig @ sig.T
            # chol_Q = jnp.linalg.cholesky(cov)
            # return mvn_logpdf(ys[0], x, chol_Q, constant=False) + M0_logpdf(x[-1])

        @partial(jax.vmap, in_axes=(0, 0, None))
        def Gamma_t(x_t_m_1, x_t, params):
            y_t, t, dt = params
            return log_potential(x_t, x_t_m_1, y_t, drift, diffusion, t, dt)

    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'guided'")

    M0 = M0_rvs, M0_logpdf
    Mt = Mt_rvs, Mt_logpdf, (ys[1:], ts[1:], dts[1:])
    Gamma_t_plus_params = Gamma_t, (ys[1:], ts[1:], dts[1:])

    kernel = lambda key, state, *_: csmc.forward_pass(key, state[0], state[1], M0, Gamma_0, Mt, Gamma_t_plus_params, N=N,
                                                **kwargs)
    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    return kernel, init



def get_pcn_csmc_kernel(ys, sigma, N, style="filtering", stop_gradient=False, **kwargs):
    pass

