"""
Gueant-style guided cSMC kernel for the corporate-bond model.
"""

from typing import Callable, Union, Any

import jax
from jax import Array
from jax.random import PRNGKey

import jax.random as jr
import jax.numpy as jnp

from jax.tree_util import tree_map
from jax.scipy.stats import norm
from jax.scipy.linalg import solve_triangular

from cd_ssm.t_csmc import backward_sampling_pass, backward_scanning_pass
from cd_ssm.utils.resamplings import normalize
from cd_ssm.utils.math import mvn_logpdf


def kernel(
        key: PRNGKey,
        x_star,
        b_star: Array,
        M_0: tuple[Callable, Callable],
        Gamma_0: Callable,
        M_t_params,
        Gamma_t: Union[Callable, tuple[Callable, Any]],
        obs_logpdf: Callable,
        chol_Q_eta: Array,
        chol_R: Array,
        psi: Array,
        resampling_func: Callable,
        ancestor_move_func: Callable,
        N: int,
        backward: bool = False,
        conditional: bool = False,
):
    """
    Guéant guided cSMC kernel.

    N is the number of non-reference particles. The internal particle count is N + 1.
    """
    ###############################
    #        HOUSEKEEPING         #
    ###############################
    z_star, eta_star = x_star
    T, D = z_star.shape

    keys = jr.split(key, T + 1)
    key_init = keys[0]
    key_backward = keys[1]
    keys_forward = keys[2:]        # length T - 1

    Gamma_t, Gamma_params = Gamma_t if isinstance(Gamma_t, tuple) else (Gamma_t, None)
    M_0_rvs, M_0_logpdf = M_0

    ###########################################
    #       Guided proposal functions         #
    ###########################################

    def Mt_tilde_rvs(key, x_t_m_1, params):
        key_z, key_eta = jr.split(key)

        # unpack
        obs_t, F_t, chol_B_t, dt = params
        z_t_m_1, eta_t_m_1 = x_t_m_1
        N = z_t_m_1.shape[0]

        # propose
        eps_z = jr.normal(key_z, shape=(N, D))
        z_t = z_t_m_1 @ F_t.T + eps_z @ chol_B_t.T
        eta_t = mid_price_proposal(key_eta, z_t, eta_t_m_1, obs_t, psi, chol_Q_eta, chol_R, dt)
        return (z_t, eta_t)

    def log_Mt_tilde(x_t_m_1, x_t, params):
        return Mt_tilde_logpdf(x_t_m_1, x_t, params, obs_logpdf, chol_Q_eta, chol_R, psi)

    #################################
    #        Initialisation         #
    #################################
    x0 = M_0_rvs(key_init, N + 1)
    if conditional:
        x0 = tree_map(lambda x0_, xs0_: x0_.at[b_star[0]].set(xs0_), x0, tree_map(lambda x: x[0], x_star))

    # Compute initial weights and normalize
    log_w0 = Gamma_0(x0) - M_0_logpdf(x0)
    log_w0 = normalize(log_w0, log_space=True)
    w0 = jnp.exp(log_w0)

    #################################
    #        Forward pass           #
    #################################

    def body(carry, inp):
        w_t_m_1, x_t_m_1 = carry
        M_t_params, Gamma_params_t, x_star_t, b_star_t_m_1, b_star_t, key_t = inp

        key_proposal_t, key_resampling_t = jax.random.split(key_t, 2)

        # Conditional resampling
        A_t = resampling_func(key_resampling_t, w_t_m_1, b_star_t_m_1, b_star_t, conditional)
        x_t_m_1 = tree_map(lambda x: jnp.take(x, A_t, axis=0), x_t_m_1)

        # Sample proposal
        x_t = Mt_tilde_rvs(key_proposal_t, x_t_m_1, M_t_params)
        if conditional:
            x_t = tree_map(lambda xt_, xs_t_: xt_.at[b_star_t].set(xs_t_), x_t, x_star_t)

        log_w_t = Gamma_t(x_t_m_1, x_t, Gamma_params_t) - log_Mt_tilde(x_t_m_1, x_t, M_t_params)
        log_w_t = normalize(log_w_t, log_space=True)
        w_t = jnp.exp(log_w_t)

        # Return next step
        next_carry = w_t, x_t
        save = log_w_t, A_t, x_t

        return next_carry, save

    inputs = (M_t_params, Gamma_params, tree_map(lambda x: x[1:], x_star), b_star[:-1], b_star[1:], keys_forward)
    _, (log_ws, As, xs) = jax.lax.scan(body, (w0, x0), inputs)

    log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)
    xs = tree_map(lambda xs_, x0_: jnp.insert(xs_, 0, x0_, axis=0), xs, x0)

    if backward:
        xs, Bs = backward_sampling_pass(key_backward, Gamma_t, Gamma_params, b_star[-1], xs, log_ws, ancestor_move_func)
    else:
        xs, Bs = backward_scanning_pass(key_backward, As, b_star[-1], xs, log_ws[-1], ancestor_move_func)

    return xs, Bs, log_ws


def _obs_var(chol_R: Array, bond_idx: Array):
    """
    Returns observation-noise variance for bond_idx.

    chol_R may be either:
        (D,)    vector of observation standard deviations
        (D, D)  Cholesky factor of observation covariance
    """
    if chol_R.ndim == 1:
        return chol_R[bond_idx] ** 2

    R = chol_R @ chol_R.T
    return R[bond_idx, bond_idx]


def _logdiffexp(a: Array, b: Array):
    """
    Computes log(exp(a) - exp(b)), assuming a >= b.
    """
    return a + jnp.log1p(-jnp.exp(b - a))


def predictive_obs_logpdf(
        z_t: Array,
        eta_t_m_1: Array,
        obs_t: tuple[Array],
        psi: Array,
        chol_Q_eta: Array,
        chol_R: Array,
        dt: Array,
):
    """
    Guéant Step-2 predictive log-weight:

        log p(obs_t | z_t, eta_{t-1})

    after integrating out eta_t and the observation noise.
    """
    bond_idx, event_type, alpha_i, obs_value = obs_t
    bond_idx = bond_idx.astype(jnp.int32)
    event_type = event_type.astype(jnp.int32)

    # variance
    Q = chol_Q_eta @ chol_Q_eta.T
    var_i = Q[bond_idx, bond_idx]
    var_eps = _obs_var(chol_R, bond_idx)
    var_tilde = var_i * dt + var_eps
    std_tilde = jnp.sqrt(var_tilde)

    # extraction
    eta_prev_i = eta_t_m_1[:, bond_idx]
    half_spread = psi[bond_idx] * jnp.exp(z_t[:, bond_idx])

    # case based evaluation
    case_0 = lambda: norm.logpdf(obs_value + half_spread, loc=eta_prev_i, scale=std_tilde)
    case_1 = lambda: norm.logpdf(obs_value - half_spread, loc=eta_prev_i, scale=std_tilde)
    case_2 = lambda: norm.logcdf(eta_prev_i - (obs_value + half_spread), loc=0.0, scale=std_tilde)
    case_3 = lambda: norm.logcdf((obs_value - half_spread) - eta_prev_i, loc=0.0, scale=std_tilde)
    def case_4():
        log_hi = norm.logcdf((obs_value + alpha_i) - eta_prev_i, loc=0.0, scale=std_tilde)
        log_lo = norm.logcdf((obs_value - alpha_i) - eta_prev_i, loc=0.0, scale=std_tilde)
        return _logdiffexp(log_hi, log_lo)
    return jax.lax.switch(event_type, [case_0, case_1, case_2, case_3, case_4])


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
    Observation-guided proposal for eta_t.

    Parameters
    ----------
    z:        (N, D)
    eta_prev:(N, D)
    obs:     Tuple (bond_idx, event_type, alpha_i, obs_value)

    Returns
    -------
    eta:     (N, D)
    """
    # house keeping
    N, D = z.shape
    key_i, key_not_i, key_tilde = jr.split(key, 3)

    bond_idx, event_type, alpha_i, obs_value = obs
    bond_idx = bond_idx.astype(jnp.int32)
    event_type = event_type.astype(jnp.int32)

    # variance
    Q = chol_Q_eta @ chol_Q_eta.T
    var_i = Q[bond_idx, bond_idx]
    var_eps = _obs_var(chol_R, bond_idx)
    var_tilde = var_i * dt + var_eps
    std_tilde = jnp.sqrt(var_tilde)

    # extraction
    eta_prev_i = eta_prev[:, bond_idx]
    half_spread = psi[bond_idx] * jnp.exp(z[:, bond_idx])

    standardise = lambda x: (x - eta_prev_i) / std_tilde

    # eta_i_tilde = eta_i + eps_i
    case_0 = lambda: obs_value + half_spread
    case_1 = lambda: obs_value - half_spread
    case_2 = lambda: eta_prev_i + std_tilde * jr.truncated_normal(
        key_tilde,
        lower=standardise(obs_value + half_spread),
        upper=jnp.inf,
        shape=(N,),
    )
    case_3 = lambda: eta_prev_i + std_tilde * jr.truncated_normal(
        key_tilde,
        lower=-jnp.inf,
        upper=standardise(obs_value - half_spread),
        shape=(N,),
    )
    case_4 = lambda: eta_prev_i + std_tilde * jr.truncated_normal(
        key_tilde,
        lower=standardise(obs_value - alpha_i),
        upper=standardise(obs_value + alpha_i),
        shape=(N,),
    )

    eta_i_tilde = jax.lax.switch(
        event_type,
        [case_0, case_1, case_2, case_3, case_4],
    )

    # eta_i | eta_i + eps_i, eta_{i,t-1}
    mean_i = (var_i * dt * eta_i_tilde + var_eps * eta_prev_i) / var_tilde
    var_post_i = (var_i * dt * var_eps) / var_tilde
    eta_i = mean_i + jnp.sqrt(var_post_i) * jr.normal(key_i, shape=(N,))

    # eta_{-i} | eta_i under eta_t | eta_{t-1} ~ N(eta_{t-1}, dt Q)
    eps = jnp.sqrt(dt) * (jr.normal(key_not_i, shape=(N, D)) @ chol_Q_eta.T)
    eps_i = eps[:, bond_idx]
    delta_i = eta_i - eta_prev[:, bond_idx]
    beta = Q[:, bond_idx] / var_i
    eta = eta_prev + (delta_i[:, None] * beta[None, :]) + eps - (eps_i[:, None] * beta[None, :])
    eta = eta.at[:, bond_idx].set(eta_i)

    return eta


def Mt_tilde_logpdf(
        x_t_m_1,
        x_t,
        params,
        observation_logpdf: Callable,
        chol_Q_eta: Array,
        chol_R: Array,
        psi: Array,
):
    """
    Log-density of the actual guided proposal:

        q_tilde(z_t, eta_t | z_{t-1}, eta_{t-1}, obs_t)

    This includes the eta prior transition term. That term cancels in the
    forward importance weight, but it is part of the proposal density.
    """
    z_t_m_1, eta_t_m_1 = x_t_m_1
    z_t, eta_t = x_t
    D = z_t.shape[-1]

    obs_t, F_t, chol_B_t, dt = params

    # inverse cholesky factors
    inv_chol_B = solve_triangular(chol_B_t,jnp.eye(D), lower=True)
    inv_chol_eta_dt = solve_triangular(jnp.sqrt(dt) * chol_Q_eta, jnp.eye(D), lower=True)

    # prior log pdfs
    log_q_z = mvn_logpdf(z_t, z_t_m_1 @ F_t.T, None, chol_inv=inv_chol_B, constant=True)
    log_prior_eta = mvn_logpdf(eta_t, eta_t_m_1, None, chol_inv=inv_chol_eta_dt, constant=True)
    log_g = observation_logpdf(z_t, eta_t, obs_t, psi, chol_R)

    # guided correction
    log_pred = predictive_obs_logpdf(z_t,eta_t_m_1, obs_t, psi, chol_Q_eta, chol_R, dt)

    return log_q_z + log_prior_eta + log_g - log_pred