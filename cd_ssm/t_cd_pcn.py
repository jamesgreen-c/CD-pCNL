"""
Implements the CD-pCN kernel.
"""
from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp
from jax.scipy.stats import norm
from jax.tree_util import tree_map, tree_reduce

from cd_ssm.t_csmc import backward_sampling_pass, backward_scanning_pass
from cd_ssm.utils.resamplings import normalize
from cd_ssm import pcn


def kernel(
    key: PRNGKey,
    x_star: Array,
    b_star: Array,
    Gamma_0: Union[Callable, tuple[Callable, Any]],
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

    keys_u = jax.random.split(key_u, 2*T)
    keys_proposals_u, keys_aux_u = keys_u[:T], keys_u[T:]

    # Unpack Gamma function
    Gamma_0, Gamma__0_params = Gamma_0 if isinstance(Gamma_0, tuple) else (Gamma_0, None)
    Gamma_t, Gamma_params = Gamma_t if isinstance(Gamma_t, tuple) else (Gamma_t, None)

    ########################################
    #         Augmented potential          #
    ########################################
    vec_norm_logpdf = jnp.vectorize(norm.logpdf, signature="(d),(d)->()", excluded=(2,))
    def Gamma_t_tilde(x_t_m_1, x_t, params):
        ell_t, rho_t, aux_u_t, aux_e_t, orig_params = params
        dt = orig_params[2]
        u_t, e_t = x_t

        val = Gamma_t(x_t_m_1, x_t, orig_params)
        val += pcn.logpdf(u_t, aux_u_t, rho_t, dt)
        val += -jnp.sum((e_t - aux_e_t) ** 2, axis=-1) / ell_t
        return val

    ########################################
    #         Auxiliary proposals          #
    ########################################

    # auxiliary endpoint proposals
    ells = jnp.broadcast_to(jnp.atleast_1d(ells), (T,))
    aux_e_std_devs = jnp.sqrt(0.5 * ells)
    aux_es = e_star + jax.random.normal(key_aux_e, shape=(T, d_x)) * aux_e_std_devs[:, None] # aux_e_t = e_star_t + N(0, 0.5 * ell_t * I)
    eps_es = jax.random.normal(key_proposals_e, shape=(T, N + 1, d_x))
    es = aux_es[:, None, :] + aux_e_std_devs[:, None, None] * eps_es                         # e_t = aux_e_t + N(0, 0.5 * ell_t * I)

    # auxiliary Wiener proposals
    dt_0, dts = Gamma__0_params[-1], Gamma_params[-1]
    dts = jnp.insert(dts, 0, dt_0, axis=0)

    rhos = jnp.broadcast_to(jnp.atleast_1d(rho), (T,))
    vmapped_pcn_propose = jax.vmap(pcn.propose, in_axes=(0, 0, 0, 0, None))
    aux_us = vmapped_pcn_propose(keys_aux_u, u_star, rhos, dts, 1)                       # aux_u_t = rho*u_t + sqrt(1 - rho^2) * W_t()
    us = vmapped_pcn_propose(keys_proposals_u, aux_us, rhos, dts, N + 1)                 # u_t = rho*aux_u_t + sqrt(1 - rho^2) * W_t()

    # print("aux_us shape: ", aux_us.shape)
    # print("u_star shape: ", u_star.shape)
    # print("us shape: ", us.shape)

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
        Gamma_params_t, x_t, b_star_t_m_1, b_star_t, key_t = inp

        # Conditional resampling
        A_t = resampling_func(key_t, w_t_m_1, b_star_t_m_1, b_star_t)
        x_t_m_1 = tree_map(lambda x: jnp.take(x, A_t, axis=0), x_t_m_1)

        log_w_t = Gamma_t(x_t_m_1, x_t, Gamma_params_t)
        log_w_t = normalize(log_w_t, log_space=True)
        w_t = jnp.exp(log_w_t)

        # Return next step
        next_carry = w_t, x_t
        save = log_w_t, A_t

        return next_carry, save

    # Run forward cSMC
    inputs = (Gamma_params, tree_map(lambda x: x[1:], xs), b_star[:-1], b_star[1:], keys_resampling,)
    _, (log_ws, As) = jax.lax.scan(body,
                                  (w0, x0),
                                  inputs,
                                )

    # Insert initial weight and particles
    log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)

    #################################
    #        Backward pass          #
    #################################
    Gamma_tilde_params = ells[1:], rhos[1:], aux_us[1:], aux_es[1:], Gamma_params
    if backward:
        xs, Bs = backward_sampling_pass(key_backward, Gamma_t_tilde, Gamma_tilde_params, b_star[-1], xs, log_ws,
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


