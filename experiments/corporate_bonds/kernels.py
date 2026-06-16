from enum import Enum
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from jax.random import PRNGKey
from jax import Array

import numpy as np
from jax.scipy.stats import norm
from jax.scipy.linalg import solve_triangular

from experiments.corporate_bonds.model import log_potential, observation_logpdf, ou_diag_transition

from cd_ssm.utils.math import mvn_logpdf
from cd_ssm.utils.mcmc_utils import aux_sampling_routine, delta_adaptation_routine
from cd_ssm import euler
from cd_ssm import brownian as br
from cd_ssm import t_csmc
from cd_ssm import t_cd_pcn
from cd_ssm import t_cd_pcnl
from cd_ssm import t_mala_af
from cd_ssm import rw_csmc
from cd_ssm import bridge
from cd_ssm import adaptation as adpt
from cd_ssm import gueant as gueant_csmc


class KernelType(Enum):
    GUEANT = 0
    PCN = 1
    RW_CSMC = 2
    PCNL = 3
    MALA_CSMC = 4

    @property
    def kernel_maker(self):
        if self == KernelType.GUEANT:
            return get_gueant_csmc_kernel
        else:
            raise NotImplementedError

    @property
    def is_random_walk(self):
        if self == KernelType.RW_CSMC or self == KernelType.MALA_CSMC:
            return True
        return False
    
    def shape_delta(self, delta, T):
        if self == KernelType.GUEANT:
            return delta
        elif self == KernelType.PCN:
            return delta * np.ones((T,))
        elif self == KernelType.PCNL:
            return delta * np.ones((T,))
        elif self == KernelType.RW_CSMC:
            return delta * np.ones((T,))
        elif self == KernelType.MALA_CSMC:
            return delta * np.ones((T,))
        else:
            return NotImplementedError("Shape delta not implemented for kernel type")

    def shared_delta(self):
        if self == KernelType.GUEANT:
            return True
        elif self == KernelType.PCN:
            return True
        elif self == KernelType.PCNL:
            return True
        elif self == KernelType.RW_CSMC:
            return True
        elif self == KernelType.MALA_CSMC:
            return True
        else:
            return NotImplementedError("Shared delta not implemented for kernel type")
        
    def shared_rho(self):
        if self == KernelType.GUEANT:
            return True
        elif self == KernelType.PCN:
            return True
        elif self == KernelType.PCNL:
            return True
        elif self == KernelType.RW_CSMC:
            return True
        elif self == KernelType.MALA_CSMC:
            return True
        else:
            return NotImplementedError("Shared rho not implemented for kernel type")
        

#######################
#   guided proposal   #
######################

def mid_price_proposal(
        key: PRNGKey, 
        z: Array,
        eta_prev: Array,
        obs: tuple[Array], 
        psi: Array,
        chol_Q_eta: Array,
        chol_R: Array,
        dt: Array,
    ):
    """
    
    """
    N, D = z.shape
    key_i, key_not_i, key_tilde = jr.split(key, 3)

    # unpacking
    bond_idx, event_type, alpha_i, obs_value = obs
    bond_idx = bond_idx.astype(jnp.int32)
    event_type = event_type.astype(jnp.int32)

    eta_p_i = eta_prev[:, bond_idx]

    # calculate half bid-ask spread
    half_spread = psi[bond_idx] * jnp.exp(z[:, bond_idx])

    # variance
    Q = chol_Q_eta @ chol_Q_eta.T
    R = chol_R @ chol_R.T
    var_i = Q[bond_idx, bond_idx]
    var_eps = R[bond_idx, bond_idx]
    var_tilde = var_i * dt + var_eps 
    std_tilde = jnp.sqrt(var_tilde)

    # sample auxiliary noisy mid-price eta_i + eps_i
    standardise = lambda x: (x - eta_p_i) / std_tilde

    case_0 = lambda: obs_value + half_spread
    case_1 = lambda: obs_value - half_spread
    case_2 = lambda: eta_p_i + std_tilde * jr.truncated_normal(
        key_tilde,
        lower=standardise(obs_value + half_spread),
        upper=jnp.inf,
        shape=(N,),
    )
    case_3 = lambda: eta_p_i + std_tilde * jr.truncated_normal(
        key_tilde,
        lower=-jnp.inf,
        upper=standardise(obs_value - half_spread),
        shape=(N,),
    )
    case_4 = lambda: eta_p_i + std_tilde * jr.truncated_normal(
        key_tilde,
        lower=standardise(obs_value - alpha_i), 
        upper=standardise(obs_value + alpha_i),
        shape=(N,))

    eta_i_tilde = jax.lax.switch(event_type, [case_0, case_1, case_2, case_3, case_4])

    # sample observed bond mid-price
    mean = (var_i * dt * eta_i_tilde + var_eps * eta_p_i) / var_tilde
    var = (var_i * dt * var_eps) / var_tilde
    std = jnp.sqrt(var)
    eta_i = mean + std * jr.normal(key_i, shape=(N,))

    # conditionally sample other mid-prices
    eps = jnp.sqrt(dt) * (jr.normal(key_not_i, shape=(N, D)) @ chol_Q_eta.T)
    eps_i = eps[:, bond_idx]
    delta_i = eta_i - eta_prev[:, bond_idx]
    beta = Q[:, bond_idx] / var_i
    eta = eta_prev + (delta_i[:, None] * beta[None, :]) + eps - (eps_i[:, None] * beta[None, :])
    eta = eta.at[:, bond_idx].set(eta_i)

    return eta

#######################
# Kernel constructors #
#######################

def get_gueant_csmc_kernel(
        obs, 
        A: Array, 
        psi: Array,
        chol_P0_z: Array,
        chol_P0_eta: Array,
        chol_Q_z: Array, 
        chol_Q_eta: Array, 
        chol_R: Array, 
        N, 
        dts, 
        style="guided", 
        **kwargs
    ):
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
    ys = obs[-1]
    T = ys.shape[0]
    D = A.shape[0]
    ts = jnp.cumsum(dts)  # dts = [0.0, dt, 2 * dt, ...]

    # precompute exact transition parameters
    z_Fs, z_chol_Bs = jax.vmap(lambda dt: ou_diag_transition(A, chol_Q_z, dt))(dts)
    inv_chol_P0_z = solve_triangular(chol_P0_z, jnp.eye(D), lower=True)
    inv_chol_P0_eta = solve_triangular(chol_P0_eta, jnp.eye(D), lower=True)

    # premap over obs tuples
    obs_0 = jax.tree_util.tree_map(lambda x: x[0], obs)
    obs_ts = jax.tree_util.tree_map(lambda x: x[1:], obs)

    if style == "guided":

        def M0_rvs(key, _):
            key, subkey = jr.split(key)
            z_0 = jr.normal(key, (N, D)) @ chol_P0_z.T
            eta_0 = jr.normal(subkey, (N, D)) @ chol_P0_eta.T
            return (z_0, eta_0)

        def M0_logpdf(x):
            z, eta = x
            val = mvn_logpdf(z, jnp.zeros_like(z), None, chol_inv=inv_chol_P0_z)
            val += mvn_logpdf(eta, jnp.zeros_like(eta), None, chol_inv=inv_chol_P0_eta)
            return val
        
        def Mt_logpdf(x_t_m_1, x_t, params):
            z_t_m_1, eta_t_m_1 = x_t_m_1
            z_t, eta_t = x_t
            _, F_t, chol_B_t, dt = params

            # calculate inverse cholesky factors
            inv_chol_B = solve_triangular(chol_B_t, jnp.eye(D), lower=True)
            inv_chol_Q_eta = solve_triangular(jnp.sqrt(dt) * chol_Q_eta, jnp.eye(D), lower=True)

            # spread and YtB evaluation
            val = mvn_logpdf(z_t, z_t_m_1 @ F_t.T, None, chol_inv=inv_chol_B, constant=False)
            val += mvn_logpdf(eta_t, eta_t_m_1, None, chol_inv=inv_chol_Q_eta, constant=False)
            return val
        
        def Gamma_0(x):
            z, eta = x
            return observation_logpdf(z, eta, obs_0, psi, chol_R) + M0_logpdf(x)

        def Gamma_t(x_t_m_1, x_t, params):
            obs_t = params[0]
            z_t, eta_t = x_t
            return observation_logpdf(z_t, eta_t, obs_t, psi, chol_R) + Mt_logpdf(x_t_m_1, x_t, params)
        
    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'guided'")

    M0 = M0_rvs, M0_logpdf
    Mt_params = (obs_ts, z_Fs[1:], z_chol_Bs[1:], dts[1:])
    Gamma_t_plus_params = Gamma_t, (obs_ts, z_Fs[1:], z_chol_Bs[1:], dts[1:])

    kernel = lambda key, state, *_: gueant_csmc.kernel(
        key, state[0], state[1], 
        M0, Gamma_0, 
        Mt_params, Gamma_t_plus_params,
        observation_logpdf, chol_Q_eta, chol_R, psi, 
        N=N, **kwargs)
    init = lambda x: (x, jnp.zeros((T,), dtype=int))

    def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
        return aux_sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)

    def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta, initial_rho, n_steps, **kwargs):
        return adpt.delta_rho_adaptation_routine(key, state[0], state[1], 
                                                kernel_, 
                                                target_acceptance,
                                                initial_delta, initial_rho,
                                                n_steps,
                                                **kwargs)

    return kernel, init, adaptation_routine, sampling_routine_fn

