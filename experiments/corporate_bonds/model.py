"""
Continuous-time corporate bond mid-YtB and spread model.

Latent state:
    s_t = (u_t, z_t)

where
    u_t: mid-YtB process
    z_t: log half-spread process
    psi_t = Psi * exp(z_t)

Observation encoding:
    obs = [value, bond_idx, event_type, alpha]

event_type:
    0: client buys from dealer D       Y = u_i - psi_i + eps
    1: client sells to dealer D        Y = u_i + psi_i + eps
    2: traded-away client buy RFQ      observed Z, with u_i - psi_i + eps >= Z
    3: traded-away client sell RFQ     observed Z, with u_i + psi_i + eps <= Z
    4: D2D trade                       observed Y in [u_i - alpha_i + eps, u_i + alpha_i + eps]
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm
from chex import PRNGKey, Array

from cd_ssm.utils.math import logdet, mvn_logpdf
from cd_ssm.delyon_hu import delyonhu
from cd_ssm import brownian


def ou_diag_transition(A, chol_Q, dt):
    """
    Exact OU transition for dz = -diag(A_diag) z dt + chol_Q dB_t.

    Parameters
    ----------
    A:       (D, D) Diagonal transition matrix
    chol_Q:  Cholesky factor of the covariance
    
    Returns
    -------
    F:   (D, D) transition matrix
    Cov: (D, D) transition covariance
    """
    Q = chol_Q @ chol_Q.T

    A_diag = jnp.diag(A)
    a_sum = A_diag[:, None] + A_diag[None, :]
    factor = jnp.where(
        jnp.abs(a_sum) > 1e-10,
        (1.0 - jnp.exp(-a_sum * dt)) / a_sum,
        dt,
    )

    F = jnp.diag(jnp.exp(-A_diag * dt))
    Cov = factor * Q
    chol_Cov = jnp.linalg.cholesky(Cov)
    return F, chol_Cov


def _diag_or_vector_at(chol_R: Array, i: Array):
    """
    Returns the scalar observation standard deviation for bond i.

    Parameters
    ----------
    chol_R: (dim,) or (dim, dim)
    """
    if chol_R.ndim == 1:
        return chol_R[i]
    elif chol_R.ndim == 2:
        return chol_R[i, i]
    else:
        raise ValueError("chol_R must have shape (dim,) or (dim, dim).")

def _logdiffexp(a: Array, b: Array):
    """ Computes log(exp(a) - exp(b)), assuming a >= b. """
    return a + jnp.log1p(-jnp.exp(b - a))

def emission(
        key: PRNGKey,
        z: Array,
        eta: Array,
        psi: Array,
        chol_R: Array,
        alpha: Array,
        bond_idx: Array,
        event_type: Array,
):
    """
    Simulates one corporate-bond event observation.

    Parameters
    ----------
    key:        PRNGKey
    eta:        (dim,) Mid-YtB state
    z:          (dim,) Log half-spread state
    psi:        (dim,) Baseline half-spread scale Psi
    chol_R:     (dim,) or (dim, dim) Observation noise standard deviations
    alpha:      (dim,) D2D interval half-widths
    bond_idx:   Integer bond index
    event_type: Integer event type in {0, 1, 2, 3, 4}

    Returns
    -------
    obs_value: Scalar observation value.

    Notes
    -----
    For event types 2 and 3, obs_value is the dealer quote Z.
    """

    key_eps, key_aux = jax.random.split(key)

    i = bond_idx
    r = _diag_or_vector_at(chol_R, i)

    spread_i = psi[i] * jnp.exp(z[i])
    eps = r * jax.random.normal(key_eps)

    done_buy = eta[i] - spread_i + eps
    done_sell = eta[i] + spread_i + eps

    # for traded-away events we simulate a quote Z consistent with the event.
    margin = jnp.abs(r * jax.random.normal(key_aux))

    # observation cases
    case_0 = lambda: done_buy                                                # client buys from dealer D
    case_1 = lambda: done_sell                                               # client sells to dealer D
    case_2 = lambda: done_buy - margin                                       # client buys from another dealer
    case_3 = lambda: done_sell + margin                                      # client sells to another dealer
    case_4 = lambda: eta[i] + eps + jr.uniform(key_aux,
                                               shape=(),
                                               minval=-alpha[i],
                                               maxval= alpha[i])             # D2D trade: observed Y lies inside an interval around u_i + eps

    return jax.lax.switch(event_type, [case_0, case_1, case_2, case_3, case_4])


# dynamics functions
def get_half_spread_dynamics(A: Array, chol_Q_z: Array):
    def drift(t, z):
        return -A @ z
    def diffusion(t, z):
        return chol_Q_z
    return drift, diffusion


def get_ytb_dynamics(chol_Q_eta: Array):    
    def drift(t, eta):
        return jnp.zeros_like(eta)
    def diffusion(t, eta):
        return chol_Q_eta
    return drift, diffusion


@partial(jax.jit, static_argnums=(1, 11,))
def get_data(
        key: PRNGKey,
        dim: int,
        dts: Array,
        A: Array,
        psi: Array,
        chol_P0_z: Array,
        chol_P0_eta: Array,
        chol_Q_z: Array,
        chol_Q_eta: Array,
        chol_R: Array,
        alpha: Array,
        sparsity_factor: float = 10.0,
):
    """
    Simulates corporate-bond latent states and sparse event observations.

    Parameters
    ----------
    key:             PRNGKey
    dim:             Number of bonds
    dts:             (K,) Time increments
    A:               (dim, dim) Discrete-time transition matrix for z
    psi:             (dim,) Baseline half-spread scale
    chol_P0_z:       (dim, dim) Initial Cholesky factor for z_0
    chol_P0_eta:     (dim, dim) Initial Cholesky factor for eta_0
    chol_Q_z:        (dim, dim) Transition Cholesky factor for z
    chol_Q_eta:      (dim, dim) Transition Cholesky factor for eta
    chol_R:          (dim,) or (dim, dim) Observation noise standard deviations
    alpha:           (dim,) D2D interval half-widths
    sparsity_factor: Observation frequency ratio between non-final bonds and final bond.

    Returns
    -------
    xs:   Tuple (zs, etas)
            zs:   (K, dim)
            etas: (K, dim)

    obs:  Tuple (bond_idxs, event_types, alphas, obs_values)
    """

    init_key, event_key, sampling_key = jax.random.split(key, 3)
    K = dts.shape[0]

    init_key_z, init_key_eta = jax.random.split(init_key)

    z0 = chol_P0_z @ jax.random.normal(init_key_z, (dim,))
    eta0 = chol_P0_eta @ jax.random.normal(init_key_eta, (dim,))

    key_bond, key_type, key_y = jax.random.split(event_key, 3)

    bond_weights = jnp.ones((dim,))
    bond_weights = bond_weights.at[:-1].set(sparsity_factor)
    bond_probs = bond_weights / jnp.sum(bond_weights)
    bond_idxs = jax.random.categorical(key_bond, jnp.log(bond_probs), shape=(K,)).astype(jnp.int32)

    event_types = jax.random.randint(key_type, (K,), minval=0, maxval=5)
    keys_y = jax.random.split(key_y, K)

    z_Fs, z_chol_Bs = jax.vmap(lambda dt: ou_diag_transition(A, chol_Q_z, dt))(dts)
    eps_zs, eps_etas = jax.random.normal(sampling_key, (2, K, dim))

    def body(carry, inps):
        z_k, eta_k = carry
        dt, F, chol_B, eps_z, eps_eta, key_y_k, bond_idx, event_type = inps

        # sample next latent state
        z_kp1 = z_k @ F.T + eps_z @ chol_B.T
        eta_kp1 = eta_k + jnp.sqrt(dt) * (eps_eta @ chol_Q_eta.T)
        x_kp1 = (z_kp1, eta_kp1)

        # sample observation
        obs_value = emission(key_y_k, z_kp1, eta_kp1, psi, chol_R, alpha, bond_idx, event_type)
        obs_k = (bond_idx, event_type, alpha[bond_idx], obs_value)

        return x_kp1, (x_kp1, obs_k)

    carry0 = (z0, eta0)
    inps = (dts, z_Fs, z_chol_Bs, eps_zs, eps_etas, keys_y, bond_idxs, event_types)
    _, (xs, obs) = jax.lax.scan(body, carry0, inps)
    return xs, obs


@partial(jnp.vectorize, signature="(d),(d)->()", excluded=(2, 3, 4))
def observation_logpdf(
        z,
        eta,
        obs,
        psi: Array,
        chol_R: Array,
):
    """
    Corporate-bond event log-likelihood.

    Parameters
    ----------
    x:      Tuple (eta, z) where
               - eta: (dim,) Mid-YtB state
               - z: (dim,) Log half-spread state
    obs:    Tuple (bond_idx, event_type, alpha_i, obs_value)
    psi:    (dim,) Baseline half-spread scale
    chol_R: (dim,) or (dim, dim)
            Observation noise standard deviations

    Returns
    -------
    val: Scalar log-likelihood contribution.
    """

    bond_idx, event_type, alpha_i, obs_value = obs

    bond_idx = bond_idx.astype(jnp.int32)
    event_type = event_type.astype(jnp.int32)

    # retrieve bond-specific emission parameters
    r_i = _diag_or_vector_at(chol_R, bond_idx)
    mid_i = eta[bond_idx]
    spread_i = psi[bond_idx] * jnp.exp(z[bond_idx])

    case_0 = lambda: norm.logpdf(obs_value, loc=mid_i - spread_i, scale=r_i)          # D2C buy: Y = eta_i - psi_i + eps
    case_1 = lambda: norm.logpdf(obs_value, loc=mid_i + spread_i, scale=r_i)          # D2C sell: Y = eta_i + psi_i + eps
    case_2 = lambda: norm.logcdf((mid_i - spread_i) - obs_value, loc=0.0, scale=r_i)  # traded-away buy RFQ:  observed quote Z, condition eta_i - psi_i + eps >= Z
    case_3 = lambda: norm.logcdf(obs_value - (mid_i + spread_i), loc=0.0, scale=r_i)  # traded-away sell RFQ: observed quote Z, condition eta_i + psi_i + eps <= Z

    def case_4():
        # D2D: observed Y, condition Y in [eta_i - alpha_i + eps, eta_i + alpha_i + eps]
        lo = obs_value - mid_i - alpha_i
        hi = obs_value - mid_i + alpha_i
        log_hi = norm.logcdf(hi, loc=0.0, scale=r_i)
        log_lo = norm.logcdf(lo, loc=0.0, scale=r_i)
        return _logdiffexp(log_hi, log_lo)

    return jax.lax.switch(event_type, [case_0, case_1, case_2, case_3, case_4])


@partial(jnp.vectorize, signature="(m,d),(m,d),(d),(d)->()", excluded=(4, 5, 6, 7, 8, 9))
def log_potential(
        z_path,
        eta_path,
        z_ep,
        eta_ep,
        obs,
        drift: tuple[Callable, Callable],
        diffusion: tuple[Callable, Callable],
        t: Array,
        dt: Array,
        psi: Array,
        chol_R: Array,
    ):

    # define covariance functions
    z_drift, eta_drift = drift
    z_diffusion, eta_diffusion = diffusion

    def _cov(_t, _x, _diffusion):
        sig = _diffusion(_t, _x) * jnp.eye(_x.shape[0])
        return sig @ sig.T
    
    _z_cov = lambda _t, _x: _cov(_t, _x, z_diffusion)  
    _eta_cov = lambda _t, _x: _cov(_t, _x, eta_diffusion)  

    # extract endpoints
    z_e = z_path[-1]
    eta_e = eta_path[-1]

    # calculate log_potential
    val = observation_logpdf(z_e, eta_e, obs, psi, chol_R)

    # TODO change these to mvn logpdfs
    val += norm.logpdf(z_e, loc=z_ep, scale=jnp.sqrt(dt) * z_diffusion(t, z_ep)).sum()
    val += norm.logpdf(eta_e, loc=eta_ep, scale=jnp.sqrt(dt) * eta_diffusion(t, eta_ep)).sum()

    val += 0.5 * (logdet(_z_cov(t, z_ep)) - logdet(_z_cov(t + dt, z_e)))
    val += 0.5 * (logdet(_eta_cov(t, eta_ep)) - logdet(_eta_cov(t + dt, eta_e)))

    val += delyonhu(z_path, z_drift, z_diffusion, t, dt)
    val += delyonhu(eta_path, eta_drift, eta_diffusion, t, dt)

    return val