import jax
import jax.numpy as jnp
from chex import Array

from cd_ssm.utils.pbar import progress_bar_scan


def delta_rho_adaptation_routine(
        key,
        init_xs, init_bs,
        kernel,
        target_acceptance,
        initial_deltas,
        initial_rhos,
        n_steps,
        min_delta=1e-12,
        max_delta=1e2,
        min_rho=1e-12,
        max_rho=.99999,
        delta_min_rate=1e-2,
        rho_min_rate=1e-5,
        window_size=100,
        delta_rate=0.1,
        rho_rate=0.01,
        shared_delta: bool = False, 
        shared_rho: bool = False,
        rho_direction: int = -1,
        **_kwargs
):
    """
    Adapts two sets of tuning parameters (deltas and rhos) for an CD-pCN(L) MCMC kernel
    targeting a desired acceptance rate for each.

    Acceptance is estimated via a Monte Carlo proxy using particle ancestry bs. 

    The two parameters are adapted in sequence within each step:
      1. Delta is adapted: the kernel is run with current (delta, rho), and delta is
         updated based on the resulting empirical acceptance rates.
      2. Rho is adapted: the kernel is run again from the updated state with the new
         delta, and rho is updated based on the resulting empirical acceptance rates.

    Adaptation uses a diminishing step-size schedule to satisfy the diminishing
    adaptation condition required for ergodicity of adaptive MCMC.

    Parameters
    ----------
    key:                    PRNGKey. Split internally for delta and rho adaptation phases.
    init_xs:                Array. Initial particle states, expected as (init_us, init_es) where the leading
                                dimension of init_us determines T, the number of time steps / particles.
    init_bs:                Array. Initial particle labels or ancestor indices, used as a proxy for 
                                detecting accepted moves (acceptance <=> label change).
    kernel:                 MCMC kernel with signature:
                                kernel(key, state, deltas, rhos, acc_hist) -> (next_xs, next_bs, ...)
                                where state = (xs, bs).
    target_acceptance:      Desired acceptance rate for both delta and rho, e.g. 0.234 or 0.44.
    initial_deltas:         Scalar initial value for delta, broadcast to shape (T,).
    initial_rhos:           Scalar initial value for rho, broadcast to shape (T,).
    n_steps:                Number of adaptation iterations to run.
    verbose:                If True, displays a progress bar over adaptation steps. Default False.
    min_delta:              Lower bound for delta. Default 1e-12.
    max_delta:              Upper bound for delta. Default 1e2.
    min_rho:                Lower bound for rho. Default 1e-12.
    max_rho:                Upper bound for rho. Default 1.0.
    min_rate:               Minimum step size for the adaptation update, preventing the schedule from
                                decaying to zero too quickly. Default 1e-2.
    window_size:            Rolling window length over which empirical acceptance rates are computed.
                                Adaptation is suppressed until at least `window_size` steps have elapsed.
                                Default 100.
    rate:                   Initial step size scaling for the Robbins-Monro adaptation update.
                                Decayed as rate / sqrt(i+1) during the run. Default 0.1.
    **_kwargs:              Absorbs unused keyword arguments for compatibility.

    Returns
    -------
    fin_state:      tuple. Final particle state (xs, bs) after all adaptation steps.
    fin_deltas:     Array, shape (T,). Final adapted delta values.
    fin_rhos:       Array, shape (T,). Final adapted rho values.
    """
    
    init_us, init_es = init_xs
    T = init_us.shape[0]

    T_delta = 1 if shared_delta else T
    T_rho   = 1 if shared_rho   else T

    adapt_delta = lambda _i, _deltas, _bs, _next_bs, _acc_hist: adapt(
        _i, _deltas, _bs, _next_bs, _acc_hist,
        min_delta, max_delta, window_size, target_acceptance, delta_min_rate, delta_rate,
        shared=shared_delta, 
        direction=+1,
    )
    adapt_rho = lambda _i, _rhos, _bs, _next_bs, _acc_hist: adapt(
        _i, _rhos, _bs, _next_bs, _acc_hist,
        min_rho, max_rho, window_size, target_acceptance, rho_min_rate, rho_rate,
        shared=shared_rho,
        direction=rho_direction
    )

    def body(carry, inp):
        state, deltas, rhos, deltas_acc_hist, rhos_acc_hist, *_ = carry
        xs, bs = state
        i, key_d_i, key_r_i = inp

        # --- adapt delta ---
        next_xs, next_bs, *_ = kernel(key_d_i, state, deltas, rhos)
        deltas, deltas_acc_hist, deltas_acc_rates = adapt_delta(
            i, deltas, bs, next_bs, deltas_acc_hist
        )

        # --- adapt rho ---
        xs, bs, state = next_xs, next_bs, (next_xs, next_bs)
        next_xs, next_bs, *_ = kernel(key_r_i, state, deltas, rhos)
        rhos, rhos_acc_hist, rhos_acc_rates = adapt_rho(
            i, rhos, bs, next_bs, rhos_acc_hist
        )

        # diagnostics: mean acceptance rates across particles
        deltas_ar = jnp.mean(deltas_acc_rates)
        rhos_ar = jnp.mean(rhos_acc_rates)

        carry_out = (
            (next_xs, next_bs), deltas, rhos,
            deltas_acc_hist, rhos_acc_hist,
            deltas_ar, rhos_ar
        )
        return carry_out, (deltas, rhos, deltas_ar, rhos_ar)

    # initial carry
    initial_deltas = initial_deltas * jnp.ones(T_delta)
    initial_rhos = initial_rhos * jnp.ones(T_rho)

    initial_acc_hist_delta = jnp.zeros((T_delta, window_size)) * jnp.nan
    initial_acc_hist_rho = jnp.zeros((T_rho, window_size)) * jnp.nan

    init = (
        (init_xs, init_bs),
        initial_deltas, initial_rhos,
        initial_acc_hist_delta, initial_acc_hist_rho,
        jnp.mean(initial_acc_hist_delta), jnp.mean(initial_acc_hist_rho)
    )

    # --- adaptation ---
    key_d, key_r = jax.random.split(key)
    inps = jnp.arange(n_steps), jax.random.split(key_d, n_steps), jax.random.split(key_r, n_steps)
    (fin_state, fin_deltas, fin_rhos, *_), hist = jax.lax.scan(body, init, inps)
    return fin_state, fin_deltas, fin_rhos, hist


def delta_ell_adaptation_routine(
        key,
        init_xs,
        init_bs,
        kernel,
        target_acceptance,
        initial_deltas,
        initial_ells,
        n_steps,
        min_delta=1e-12,
        max_delta=1e2,
        min_ell=1e-12,
        max_ell=1e2,
        delta_min_rate=1e-2,
        ell_min_rate=1e-3,
        window_size=100,
        delta_rate=0.1,
        ell_rate=0.05,
        shared_delta: bool = False,
        shared_ell: bool = False,
        **_kwargs
):
    """
    Adapts two random-walk tuning parameters, deltas and ells, for a particle
    MCMC kernel targeting a desired acceptance rate for each.

    Acceptance is estimated using ancestry-label changes:
        accepted_t = next_bs[t] != bs[t].

    The two parameters are adapted sequentially within each iteration:
    1. Run the kernel with current (deltas, ells), then adapt deltas.
    2. Run the kernel again from the updated state, then adapt ells.

    Both deltas and ells are random-walk scale parameters. Therefore both use
    direction=+1:
        acceptance > target  => increase scale
        acceptance < target  => decrease scale

    Parameters
    ----------
    key:                    PRNGKey.
    init_xs:                Initial latent state, expected as a pytree whose first component has leading time dimension T.
    init_bs:                Initial ancestry labels, shape (T,).
    kernel:                 MCMC kernel with signature: kernel(key, state, deltas, ells) -> (next_xs, next_bs, ...), where state = (xs, bs).
    target_acceptance:      Desired acceptance rate.
    initial_deltas:         Initial delta value(s). Broadcast to shape (T,) unless shared_delta=True.
    initial_ells:           Initial ell value(s). Broadcast to shape (T,) unless shared_ell=True.
    n_steps:                Number of adaptation iterations.
    min_delta:              Lower clipping bound for deltas.
    max_delta:              Upper clipping bound for deltas.
    min_ell:                Lower clipping bound for ells.
    max_ell:                Upper clipping bound for ells.
    delta_min_rate:         Floor for the diminishing Robbins-Monro step size used to adapt deltas.
    ell_min_rate:           Floor for the diminishing Robbins-Monro step size used to adapt ells.
    window_size:            Rolling acceptance-window size.
    delta_rate:             Initial Robbins-Monro rate constant for deltas.
    ell_rate:               Initial Robbins-Monro rate constant for ells.
    shared_delta:           If True, adapt a single shared delta by pooling acceptance across all time steps.
    shared_ell:             If True, adapt a single shared ell by pooling acceptance across all time steps.
    **_kwargs:              Absorbs unused keyword arguments for compatibility.

    Returns
    -------
    fin_state:              Final state (xs, bs).
    fin_deltas:             Final adapted deltas.
    fin_ells:               Final adapted ells.
    hist:                   Tuple of adaptation histories: (deltas_hist, ells_hist, delta_acceptance_hist, ell_acceptance_hist).
    """

    init_us, _ = init_xs
    T = init_us.shape[0]

    T_delta = 1 if shared_delta else T
    T_ell = 1 if shared_ell else T

    adapt_delta = lambda _i, _deltas, _bs, _next_bs, _acc_hist: adapt(
        _i, _deltas, _bs, _next_bs, _acc_hist,
        min_delta, max_delta, window_size, target_acceptance, delta_min_rate, delta_rate, 
        shared=shared_delta,
        direction=+1,
    )

    adapt_ell = lambda _i, _ells, _bs, _next_bs, _acc_hist: adapt(
        _i, _ells, _bs, _next_bs, _acc_hist, 
        min_ell, max_ell, window_size, target_acceptance, ell_min_rate, ell_rate,
        shared=shared_ell,
        direction=+1,
    )

    def body(carry, inp):
        state, deltas, ells, deltas_acc_hist, ells_acc_hist, *_ = carry
        xs, bs = state
        i, key_delta_i, key_ell_i = inp

        # --- adapt delta ---
        next_xs, next_bs, *_ = kernel(key_delta_i, state, deltas, ells)
        deltas, deltas_acc_hist, deltas_acc_rates = adapt_delta(
            i, deltas, bs, next_bs, deltas_acc_hist
        )

        # --- adapt ell ---
        state = (next_xs, next_bs)
        xs, bs = state

        next_xs, next_bs, *_ = kernel(key_ell_i, state, deltas, ells)
        ells, ells_acc_hist, ells_acc_rates = adapt_ell(
            i, ells, bs, next_bs, ells_acc_hist
        )

        deltas_ar = jnp.mean(deltas_acc_rates)
        ells_ar = jnp.mean(ells_acc_rates)

        carry_out = (
            (next_xs, next_bs),
            deltas, ells,
            deltas_acc_hist, ells_acc_hist,
            deltas_ar, ells_ar
        )

        return carry_out, (deltas, ells, deltas_ar, ells_ar)

    initial_deltas = initial_deltas * jnp.ones(T_delta)
    initial_ells = initial_ells * jnp.ones(T_ell)

    initial_acc_hist_delta = jnp.zeros((T_delta, window_size)) * jnp.nan
    initial_acc_hist_ell = jnp.zeros((T_ell, window_size)) * jnp.nan

    init = (
        (init_xs, init_bs),
        initial_deltas, initial_ells,
        initial_acc_hist_delta, initial_acc_hist_ell,
        jnp.mean(initial_acc_hist_delta), jnp.mean(initial_acc_hist_ell),
    )

    key_delta, key_ell = jax.random.split(key)
    inps = (jnp.arange(n_steps), jax.random.split(key_delta, n_steps), jax.random.split(key_ell, n_steps))
    (fin_state, fin_deltas, fin_ells, *_), hist = jax.lax.scan(body, init, inps)

    return fin_state, fin_deltas, fin_ells, hist


def adapt(
        i,
        tuner: Array,
        bs: Array,
        next_bs: Array,
        accepted_history: Array,
        min_tuner,
        max_tuner,
        window_size,
        target_acceptance,
        min_rate,
        rate,
        shared: bool = False,
        direction: int = 1
):
    """
    Performs one step of Robbins-Monro stochastic approximation to adapt a
    scalar tuning parameter (per time step) toward a target acceptance rate.

    Acceptance at each time step is estimated from particle label changes in the
    SMC population.

    Acceptance events are tracked in a rolling window, and the tuner is only
    updated when:
      - At least `window_size` steps have elapsed (to ensure reliable estimates), and
      - The estimated acceptance rate deviates from the target by more than 0.05.

    The update rule is a normalised Robbins-Monro step:
        tuner <- tuner + step * tuner * (acc_rate - target) / target
    with a diminishing step size: step = max(min_rate, rate / sqrt(i+1)).

    Parameters
    ----------
    i:                  Current iteration index, used to compute the diminishing step size and
                            to suppress adaptation before `window_size` steps have elapsed.
    tuner:              Array, shape (T,). Current tuning parameter values, 
                            one per time step / particle dimension.
    bs:                 Array, shape (T,). Particle labels before the kernel step.
    next_bs:            Array, shape (T,). Particle labels after the kernel step. 
                            Difference from `bs` serves as the acceptance indicator.
    accepted_history:   Array, shape (T, window_size). Rolling buffer of past acceptance indicators. 
                            Column 0 is the most recent.
    min_tuner:          Lower clipping bound for the tuner after adaptation.
    max_tuner:          Upper clipping bound for the tuner after adaptation.
    window_size:        Length of the rolling acceptance window. Adaptation is suppressed for
                            the first `window_size` iterations.
    target_acceptance:  Desired acceptance rate (e.g. 0.234 for high-dimensional, 0.44 for 1D).
    min_rate:           Floor on the diminishing step size to prevent it collapsing too early.
    rate:               Initial step size scale factor, decayed as rate / sqrt(i+1).
    mean_axis:          The axes to mean the acceptance rate over

    Returns
    -------
    tuner:              Array, shape (T,). Updated tuning parameter values.
    accepted_history:   Array, shape (T, window_size). Updated rolling acceptance 
                            buffer with the latest observations prepended.
    acceptance_rates:   Array, shape (T,). Empirical acceptance rates over the 
                            current window, one per time step.
    """
    accepted = next_bs != bs

    # if scalar tuning parameter: pool all particles into a single acceptance signal
    if shared:
        accepted = jnp.mean(accepted)[None]

    # augment rolling window
    accepted_history = accepted_history.at[:, 1:].set(accepted_history[:, :-1])
    accepted_history = accepted_history.at[:, 0].set(accepted)
    acceptance_rates = jnp.nanmean(accepted_history, axis=1)

    # only adapt if outside tolerance band and window is fully populated with samples
    flag = jnp.abs(acceptance_rates - target_acceptance) > 0.05
    flag &= i > window_size

    # --- Robbins-Monro schedule) ---
    rate_i = jnp.maximum(min_rate, rate / (i + 1) ** 0.5)

    # --- adaptation update ---
    tuner_updated = tuner + direction * rate_i * tuner * (
        acceptance_rates - target_acceptance
    ) / target_acceptance

    tuner = jnp.where(flag, tuner_updated, tuner)
    tuner = jnp.clip(tuner, min_tuner, max_tuner)

    return tuner, accepted_history, acceptance_rates


    # initial_deltas = initial_deltas * jnp.ones(T)
    # initial_rhos = initial_rhos * jnp.ones(T)
    # initial_acc_hist = jnp.zeros((T, window_size)) * jnp.nan
    # init = (
    #     (init_xs, init_bs),
    #     initial_deltas, initial_rhos,
    #     initial_acc_hist, initial_acc_hist,
    #     jnp.mean(initial_acc_hist), jnp.mean(initial_acc_hist)
    # )