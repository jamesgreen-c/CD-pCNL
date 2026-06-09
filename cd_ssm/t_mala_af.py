from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp
from jax.scipy.stats import norm
from jax.tree_util import tree_map, tree_reduce

from cd_ssm.t_csmc import backward_sampling_pass, backward_scanning_pass
from cd_ssm.utils.resamplings import normalize
from cd_ssm import pcn
from cd_ssm import brownian


def kernel(
    key: PRNGKey,
    x_star: Array,
    b_star: Array,
    Gamma_0: Union[Callable, tuple[Callable, Any]],
    Gamma_t: Union[Callable, tuple[Callable, Any]],
    ells: Array,
    deltas: Array,
    resampling_func: Callable,
    ancestor_move_func: Callable,
    N: int,
    backward: bool = False,
):
    """
    CD-MALA kernel.

    Parameters
    ----------
    key:                Random number generator key.
    x_star:             Reference trajectory to update.
    b_star:             Indices of the reference trajectory.
    Gamma_0:            Initial weight function.
    Gamma_t:            If a tuple, the first element is the function and the second element is the parameters.
                            Gamma(None, x, None) returns the log-likelihood at time 0.
    ells:               (es) Step-size for the random walk.
    deltas:             (us) RW step-size parameters.
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

    # grad functions. Assumes argnums go order u, e
    Gamma_0_grad = jax.grad(Gamma_0, 1)
    Gamma_t_grad = jax.grad(Gamma_t, (2, 3))

    ###########################################
    #       Modified weight functions         #
    ###########################################

    def Gamma_0_tilde(x_0, ell_0, aux_e_0):
        u_0, e_0 = x_0

        val_grad_Gamma = jax.value_and_grad(Gamma_0, 1)
        vec_Gamma_val_grad = jnp.vectorize(val_grad_Gamma, signature='(m,d),(d)->(),(d)')        
        Gamma_val, e_Gamma_grad = vec_Gamma_val_grad(u_0, e_0)

        # Compute N(aux_e_t | e_t + 0.5 * delta_t * Gamma_e_grad, 0.5 * ell_t * I)
        gradient_log_pdf = -jnp.sum((aux_e_0 - e_0 - 0.5 * ell_0 * e_Gamma_grad) ** 2, axis=-1) / ell_0
        return Gamma_val + gradient_log_pdf

    def G_0_tilde(x_0, ell_0, aux_e_0):
        _, e_0 = x_0

        # Compute N(e_t | aux_e_t, 0.5 * ell_t * I), note how 0.5 / 0.5 = 1
        proposal_log_pdf = -jnp.sum((e_0 - aux_e_0) ** 2, axis=-1) / ell_0
        return Gamma_0_tilde(x_0, ell_0, aux_e_0) - proposal_log_pdf

    def Gamma_t_tilde(x_t_m_1, x_t, params):
        ell_t, rho_t, aux_u_t, aux_e_t, orig_params = params
        dt = orig_params[2]
        u_t_m_1, e_t_m_1 = x_t_m_1
        u_t, e_t = x_t

        # evaluate gamma and its gradients wrt u and e
        val_grad_Gamma = jax.value_and_grad(Gamma_t, (2, 3))
        def _flat_val_grad(up, ep, u, e, _params):
            _Gamma_val, (_u_grad, _e_grad) = val_grad_Gamma(up, ep, u, e, _params)
            return _Gamma_val, _u_grad, _e_grad
        
        vec_Gamma_val_grad = jnp.vectorize(_flat_val_grad, signature='(m,d),(d),(m,d),(d)->(),(m,d),(d)', excluded=(4,))
        Gamma_val, u_Gamma_grad, e_Gamma_grad = vec_Gamma_val_grad(u_t_m_1, e_t_m_1, u_t, e_t, orig_params)

        # Compute N(aux_e_t | e_t + 0.5 * ell_t * Gamma_grad, 0.5 * ell_t * I) and P^∇_t(aux_u_t | u_t)
        e_grad_log_pdf = -jnp.sum((aux_e_t - e_t - 0.5 * ell_t * e_Gamma_grad) ** 2, axis=-1) / ell_t
        u_grad_log_pdf = brownian.Mala.logpdf(u_t, aux_u_t, rho_t, u_Gamma_grad, dt)

        return Gamma_val + e_grad_log_pdf + u_grad_log_pdf

    def G_t_tilde(x_t_m_1, x_t, params):
        ell_t, rho_t, aux_u_t, aux_e_t, orig_params = params
        dt = orig_params[2]
        u_t, e_t = x_t

        # Compute N(e_t | aux_e_t, 0.5 * ell_t * I) and P_t(u_t | aux_u_t)
        e_prop_log_pdf = -jnp.sum((e_t - aux_e_t) ** 2, axis=-1) / ell_t
        u_prop_log_pdf = brownian.logpdf(u_t, aux_u_t, rho_t, dt)

        return Gamma_t_tilde(x_t_m_1, x_t, params) - e_prop_log_pdf - u_prop_log_pdf

    ########################################
    #         Auxiliary proposals          #
    ########################################
    
    # Compute gradients
    e_grad_log_w_star_0 = Gamma_0_grad(u_star[0], e_star[0])
    u_grad_log_w_star, e_grad_log_w_star = jax.vmap(
        Gamma_t_grad, 
        [0, 0, 0, 0, 0]
    )(u_star[:-1], e_star[:-1], u_star[1:], e_star[1:], Gamma_params)
    u_grad_log_w_star = jnp.insert(u_grad_log_w_star, 0, jnp.zeros_like(u_star[0]), axis=0)
    e_grad_log_w_star = jnp.insert(e_grad_log_w_star, 0, e_grad_log_w_star_0, axis=0)

    # auxiliary endpoint proposals
    ells = jnp.broadcast_to(jnp.atleast_1d(ells), (T,))
    aux_e_std_devs = jnp.sqrt(0.5 * ells)
    eps_aux_es = jax.random.normal(key_aux_e, shape=(T, d_x))
    aux_es = e_star + 0.5 * ells[:, None] * e_grad_log_w_star + aux_e_std_devs[:, None] * eps_aux_es
    eps_es = jax.random.normal(key_proposals_e, shape=(T, N + 1, d_x))
    es = aux_es[:, None, :] + aux_e_std_devs[:, None, None] * eps_es

    # auxiliary Wiener proposals
    dt_0, dts = Gamma__0_params[-1], Gamma_params[-1]
    dts = jnp.insert(dts, 0, dt_0, axis=0)

    deltas = jnp.broadcast_to(jnp.atleast_1d(deltas), (T,))
    brownian_proposal = jax.vmap(brownian.propose, in_axes=(0, 0, 0, 0, None))
    mala_proposal = jax.vmap(brownian.Mala.propose, in_axes=(0, 0, 0, 0, 0, None))
    aux_us = mala_proposal(keys_aux_u, u_star, deltas, u_grad_log_w_star, dts, 1)                       # aux_u_t = rho*u_t + sqrt(1 - rho^2) * W_t()
    us = brownian_proposal(keys_proposals_u, aux_us, deltas, dts, N + 1) 

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
    log_w0 = G_0_tilde(x0, ells[0], aux_es[0])
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

        log_w_t = G_t_tilde(x_t_m_1, x_t, Gamma_tilde_params_t)
        log_w_t -= jnp.max(log_w_t)
        w_t = normalize(log_w_t)

        # Return next step
        next_carry = w_t, x_t
        save = log_w_t, A_t

        return next_carry, save

    # Run forward cSMC
    Gamma_tilde_params = ells[1:], deltas[1:], aux_us[1:], aux_es[1:], Gamma_params
    inputs = (Gamma_tilde_params, tree_map(lambda x: x[1:], xs), b_star[:-1], b_star[1:], keys_resampling,)
    _, (log_ws, As) = jax.lax.scan(body,
                                  (w0, x0),
                                  inputs,
                                )

    # Insert initial weight and particle
    log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)

    #################################
    #        Backward pass          #
    #################################
    if backward:
        xs, Bs = backward_sampling_pass(key_backward, Gamma_t_tilde, Gamma_tilde_params, b_star[-1], xs, log_ws,
                                        ancestor_move_func)
    else:
        xs, Bs = backward_scanning_pass(key_backward, As, b_star[-1], xs, log_ws[-1], ancestor_move_func)
    
    # is_any_nan = tree_reduce(
    #     lambda acc, x: jnp.logical_or(acc, jnp.any(~jnp.isfinite(x))),
    #     xs,
    #     initializer=False,
    # )
    # xs = tree_map(lambda x_new, x_ref: jnp.where(is_any_nan, x_ref, x_new), xs, x_star)
    # Bs = jnp.where(is_any_nan, b_star, Bs)

    return xs, Bs, log_ws

