from enum import Enum
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.stats import norm


from experiments.lgssm.model import log_likelihood, log_potential, log_pdf
from cd_ssm.utils.numerics import euler
from cd_ssm.utils.math import mvn_logpdf
from cd_ssm import brownian as br
from cd_ssm import csmc


class KernelType(Enum):
    CSMC = 0
    PCN = 1

    @property
    def kernel_maker(self):
        if self == KernelType.CSMC:
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


def get_csmc_kernel(ys, drift: Callable, diffusion: Callable, sigma, N, style="bootstrap", **kwargs):
    """
    
    TODO:
        1. Implement a generic euler scheme:
            - This will require making model.py define drift and diffusion functions, then passing this to the euler scheme
        2. Implement a generic transform W to X scheme. This is just an euler with a predetermined noise path
        3. Implement a generic DelyonHu bridge SDE to calculate the potential function as in Stanton 2025
            - This will require an additional generic implementation of Reimann summations (numeric integral calculation)
    
        THERE IS SOME WEIRD THINGS GOING ON AT TIME 0
        NOTICE THAT IN THE BACKWARD PROPOSAL ALL ENDPOINTS CAN BE SAMPLED BEFORE RUNNING cSMC
    """

    T, dz = ys.shape

    if style == "bootstrap":

        def M0_rvs(key, _):
            """ Returns t0 distribution for pathspace not z-space """
            x0 = sigma * jr.normal(key, (N + 1, dz))
            return x0
        
        M0_logpdf = lambda x: norm.logpdf(x, scale=sigma).sum()
        M0_logpdf = jnp.vectorize(M0_logpdf, signature="(d)->()")
        
        def Mt_rvs(key, e_t_m_1, params):
            y_t, t, dt, num = params

            key1, key2 = jr.split(key)
            z_t0 = jnp.zeros((dz,))
            
            z_t = br.simulate(key1, z_t0, dt, num, N + 1)  # exact brownian simulation
            e_ts = euler(key2, drift, diffusion, e_t_m_1, t, dt, 1, N + 1)
            return z_t, e_ts[-1]
        
        def G0(x):
            """ Returns discrete weight for x0 """
            sig = diffusion(0, x)
            cov = sig @ sig.T
            chol_Q = jnp.linalg.cholesky(cov)
            return mvn_logpdf(ys[0], x, chol_Q, constant=False)
        
        def Gt(z_t, e_t_m_1, e_t, params):
            y_t = params[0]
            params = (e_t_m_1, e_t, *params[1:])
            return log_potential(z_t, y_t, drift, diffusion, params)

    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'bootstrap'")

    M0 = M0_rvs, M0_logpdf
    Mt = Mt_rvs, (ys[1:], ts[1:], dts)
    Gt_plus_params = Gt, (ys[1:], ts[1:], dts)

    kernel = lambda key, state, *_: csmc.kernel(key, state[0], state[1], M0, G0, Mt, Gt_plus_params, N=N,
                                                **kwargs)
    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    return kernel, init



def get_pcn_csmc_kernel(ys, sigma, N, style="filtering", stop_gradient=False, **kwargs):
    pass

