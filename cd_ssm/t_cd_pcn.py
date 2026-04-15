"""
Implements the CD-PCN kernel.

# TODO
# 1. Write the updates with JAX pytrees for tuples - DONE
# 2. Generate all pcn and rw auxiliary and particle proposals as in the file below before any CSMC ran
# 3. Take the Gamma function and write a Gamma_prime function that adds the necessary correction term at the bottom of this file
# 4. Use Gamma prime to do the forward and backward passes in this file
# 5. Then the pcn kernel generator should be analagous to the rw_csmc kernel generators from gradient-csmc repository, except with the Gamma functions
#     identical to the ones used by the csmc kernel generator from this repository
"""
from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_reduce

from cd_ssm.csmc import backward_sampling_pass, backward_scanning_pass
from cd_ssm.utils.resamplings import normalize
from cd_ssm import pcn


def kernel(
    key: PRNGKey,
    x_star: Array,
    b_star: Array,
    Gamma_0: Callable,
    Gamma_t: Union[Callable, tuple[Callable, Any]],
    ells: Array,
    rho: Array,
    resampling_func: Callable,
    ancestor_move_func: Callable,
    N: int,
    backward: bool = False,
):
    """
    CD-pCN kernel.

    Parameters
    ----------
    key:                Random number generator key.
    x_star:             Reference trajectory to update.
    b_star:             Indices of the reference trajectory.
    Gamma_0:            Initial weight function.
    Gamma_t:            If a tuple, the first element is the function and the second element is the parameters.
                            Gamma(None, x, None) returns the log-likelihood at time 0.
    ells:               Step-size for the random walk.
    rho:                pCN interpolation parameters.
    resampling_func:    Resampling scheme to use.
    ancestor_move_func: Function to move the last ancestor indices.
    N:                  Number of particles to use (N+1, if we include the reference trajectory).
    backward:           Whether to run the backward sampling kernel.

    Returns
    -------
    xs:                 Particles.
    bs:                 Indices of the ancestors.
    """
    ###############################
    #        HOUSEKEEPING         #
    ###############################
    u_star, e_star = x_star
    T, d_x = e_star.shape

    keys = jax.random.split(key, T + 2)
    key_e, key_u, key_backward, keys_resampling = keys[0], keys[1], keys[2], keys[3:]
    key_aux_e, key_proposals_e = jax.random.split(key_e)
    key_aux_u, key_proposals_u = jax.random.split(key_u)

    # Unpack Gamma function
    Gamma_t, Gamma_params = Gamma_t if isinstance(Gamma_t, tuple) else (Gamma_t, None)

    ###########################################
    #       Modified weight functions         #
    ###########################################
    vmapped_pcn_logpdf = jax.vmap(lambda _up, _u, _rho, _dt: pcn.logpdf(_up, _u, _rho, _dt), in_axes=(None, 0, None, None))
    def Gamma_t_tilde(x_t_m_1, x_t, params):
        u_star_t, aux_u_t, original_params_t = params
        dt = original_params_t[-1]
        u_t, _ = x_t

        val = Gamma_t(x_t_m_1, x_t, original_params_t)        
        val += pcn.logpdf(u_star_t, aux_u_t, rho, dt) - vmapped_pcn_logpdf(aux_u_t, u_t, rho, dt)
        return val
    
    ########################################
    #         Auxiliary proposals          #
    ########################################

    # auxiliary endpoint proposals
    aux_e_std_devs = jnp.sqrt(0.5 * ells)
    aux_es = e_star + jax.random.normal(key_aux_e, shape=(T, d_x)) * aux_e_std_devs[:, None] # aux_e_t = e_star_t + N(0, 0.5 * ell_t * I)
    eps_es = jax.random.normal(key_proposals_e, shape=(T, N + 1, d_x))
    es = aux_es[:, None, :] + aux_e_std_devs[:, None, None] * eps_es                         # e_t = aux_e_t + N(0, 0.5 * ell_t * I)

    # auxiliary Wiener proposals
    aux_us = pcn.propose(key_aux_u, u_star, rho, Gamma_params[-1], 1)                        # aux_u_t = rho*u_t + sqrt(1 - rho^2) * W_t()
    us = pcn.propose(key_proposals_u, aux_us, rho, Gamma_params[-1], N + 1)                  # u_t = rho*aux_u_t + sqrt(1 - rho^2) * W_t()

    # Replace the retained particle with the reference trajectory at each time t
    xs = (us, es)
    xs = tree_map(
        lambda xs_, x_star_: jax.vmap(lambda xt, bt, xst: xt.at[bt].set(xst))(xs_, b_star, x_star_),
        xs,
        x_star,
    )   

    #################################
    #        Initialisation         #
    #################################
    x0 = tree_map(lambda x: x[0], xs)

    # Compute initial weights and normalize
    log_w0 = Gamma_0(x0)
    log_w0 -= jnp.max(log_w0)
    w0 = normalize(log_w0, log_space=False)

    #################################
    #        Forward pass           #
    #################################
    def body(carry, inp):
        w_t_m_1, x_t_m_1 = carry
        Gamma_tilde_params_t, x_t, b_star_t_m_1, b_star_t, key_t = inp

        # Conditional resampling
        A_t = resampling_func(key_t, w_t_m_1, b_star_t_m_1, b_star_t)
        x_t_m_1 = tree_map(lambda x: jnp.take(x, A_t, axis=0), x_t_m_1)

        log_w_t = Gamma_t_tilde(x_t_m_1, x_t, Gamma_tilde_params_t)
        log_w_t = normalize(log_w_t, log_space=True)
        w_t = jnp.exp(log_w_t)

        # Return next step
        next_carry = w_t, x_t
        save = log_w_t, A_t

        return next_carry, save

    # Run forward cSMC
    Gamma_tilde_params = u_star, aux_us, Gamma_params
    inputs = (Gamma_tilde_params, tree_map(lambda x: x[1:], xs), b_star[:-1], b_star[1:], keys_resampling,)
    _, (log_ws, As) = jax.lax.scan(body,
                                  (w0, x0),
                                  inputs,
                                )

    # Insert initial weight and particles
    log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)

    #################################
    #        Backward pass          #
    #################################
    if backward:
        xs, Bs = backward_sampling_pass(key_backward, Gamma_t, Gamma_params, b_star[-1], xs, log_ws,
                                        ancestor_move_func)
    else:
        xs, Bs = backward_scanning_pass(key_backward, As, b_star[-1], xs, log_ws[-1], ancestor_move_func)

    is_any_nan = tree_reduce(
        lambda acc, x: jnp.logical_or(acc, jnp.any(~jnp.isfinite(x))),
        xs,
        initializer=False,
    )
    xs = tree_map(lambda x_new, x_ref: jnp.where(is_any_nan, x_ref, x_new), xs, x_star)
    Bs = jnp.where(is_any_nan, b_star, Bs)

    return xs, Bs, log_ws


