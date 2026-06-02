import argparse
import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.scipy.special import logsumexp

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from experiments.ornstein_uhlenbeck.kernels import KernelType, get_csmc_kernel
from experiments.ornstein_uhlenbeck.model import get_data, get_dynamics

from cd_ssm import bridge
from cd_ssm.utils.common import force_move, barker_move
from cd_ssm.utils.kalman import sampling, filtering
from cd_ssm.utils.resamplings import killing, multinomial
from cd_ssm.utils.timing import block_until_ready_tree

import tensorflow_probability.substrates.jax as tfp

# jax.config.update("jax_enable_x64", False)
# jax.config.update("jax_platform_name", "cpu")

# --- general config ---
MIN_DELTA = 1e-12
MAX_DELTA = 1e1
MIN_RHO = 1e-5
MAX_RHO = 1 - 1e-5
DELTA_MIN_RATE = 1e-3
RHO_MIN_RATE = 1e-6
ADAPTATION_WINDOW = 100
DELTA_ADAPTATION_RATE = 0.5
RHO_ADAPTATION_RATE = 0.025

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=10)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--K", dest="K", type=int, default=1)
parser.add_argument("--M", dest="M", type=int, default=5)

parser.add_argument("--log-var", dest="log_var", type=float, default=0)
parser.add_argument("--phi", dest="phi", type=float, default=0.8)

parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--mesh-num", dest="mesh_num", type=int, default=10)

parser.add_argument("--adaptation", dest="adaptation", type=int, default=500)
parser.add_argument("--burnin", dest="burnin", type=int, default=0)
parser.add_argument("--n-samples", dest="n_samples", type=int, default=1000)

parser.add_argument("--rho", dest="rho", type=float, default=.25)
parser.add_argument("--rho-scale", dest="rho_scale", type=float, default=1/5)
parser.add_argument("--rho-arg", dest="rho_arg", type=str, default="D")

parser.add_argument("--delta", dest="delta", type=float, default=1)
parser.add_argument("--delta-scale", dest="delta_scale", type=float, default=1)
parser.add_argument("--delta-arg", dest="delta_arg", type=str, default="D")

parser.add_argument("--target", dest="target", type=int, default=75)
parser.add_argument("--target-stat", dest='target_stat', type=str, default="mean")

parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.CSMC)
parser.add_argument("--style", dest="style", type=str, default="guided")

parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--resampling", dest='resampling', type=str, default="multinomial")
parser.add_argument("--last-step", dest='last_step', type=str, default="barker")
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

parser.add_argument("--debug", action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument("--verbose", action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=False)

parser.add_argument("--plot", action='store_true')
parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.set_defaults(plot=False)

args = parser.parse_args()

print(f"""
##################################
#        LGSSM EXPERIMENT        #
##################################
Configuration:
    - T:         {args.T}
    - kernel:    {KernelType(args.kernel).name}
    - style:     {args.style}
    - D:         {args.D}
    - N-samples  {args.n_samples}
    - Adaptation {args.adaptation}
    - Burnin     {args.burnin}
""")

# BACKEND CONFIG
NOW = time.time()

# PARAMETERS
KEY = jax.random.PRNGKey(args.seed)
ALL_KEYS = jax.random.split(KEY, args.K + 1)
WARMUP_KEY = ALL_KEYS[0]
EXPERIMENT_KEYS = ALL_KEYS[1:]

kernel_type = KernelType(args.kernel)
SHARED_DELTA = kernel_type.shared_delta()
SHARED_RHO = kernel_type.shared_rho()

# DELTA AND RHO CONFIG
if args.delta_arg == "D":
    DELTA = args.delta / args.D ** args.delta_scale
elif args.delta_arg == "T":
    DELTA = args.delta / args.T ** args.delta_scale
elif args.delta_arg == "DT" or args.delta_arg == "TD":
    DELTA = args.delta / (args.D * args.T) ** args.delta_scale
else:
    DELTA = args.delta

if kernel_type.is_random_walk:
    # overwrite rho config
    RHO = DELTA / args.mesh_num
    MIN_RHO = MIN_DELTA / args.mesh_num
    MAX_RHO = MAX_DELTA
    RHO_MIN_RATE = DELTA_MIN_RATE
    RHO_ADAPTATION_RATE = DELTA_ADAPTATION_RATE
    RHO_DIRECTION = +1

else:
    RHO_DIRECTION = -1
    rho_m1 = 1 - args.rho
    if args.rho_arg == "D":
        RHO = 1 - (rho_m1 / args.D ** args.rho_scale)
    elif args.rho_arg == "T":
        RHO = 1 - (rho_m1 / args.T ** args.rho_scale)
    elif args.rho_arg == "DT" or args.rho_arg == "TD":
        RHO = 1 - (rho_m1 / (args.D * args.T) ** args.rho_scale)
    else:
        RHO = args.rho

print(f"""
ADAPTATION CONFIG:        
    - delta init:                {DELTA}
    - min/max delta:             {MIN_DELTA}/{MAX_DELTA}
    - delta adaptation rate:     {DELTA_ADAPTATION_RATE}
    - rho init:                  {RHO}
    - min/max rho:               {MIN_RHO}/{MAX_RHO}
    - rho adaptation rate:       {RHO_ADAPTATION_RATE}
    - rho adaptation direction:  {RHO_DIRECTION}
""")


if args.resampling == "killing":
    resampling_fn = killing
elif args.resampling == "multinomial":
    resampling_fn = multinomial
else:
    raise ValueError(f"Unknown resampling {args.resampling}")

if args.last_step == "forced":
    last_step_fn = force_move
elif args.last_step == "barker":
    last_step_fn = barker_move
else:
    raise ValueError(f"Unknown last step {args.last_step}")


# --- target acceptance rate ---
TARGET_ALPHA = args.target / 100 
if args.target_stat.isnumeric():
    TARGET_STAT = float(args.target_stat) / 100
else:
    TARGET_STAT = args.target_stat

# --- dynamics config ---
SIGMA = 10 ** (args.log_var / 2)
PHI = args.phi
DTs = jnp.repeat(args.T / args.steps, args.steps)
Ts = jnp.cumsum(DTs)
DRIFT, DIFFUSION = get_dynamics(PHI, SIGMA)
OBS_SIGMA = 0.2

# --- path reconstructor ---
@jax.vmap
def to_path(us, es):
    """
    us: (n_samples, M, steps, mesh, D)
    es: (n_samples, M, steps, D)
    """
    @jax.vmap
    def _to_path(_us, _es):
        """
        _us: (M, steps, mesh, D)
        _es: (M, steps, D)
        """
        _reconstructor = jax.vmap(lambda u, ep, e, t, dt: bridge.to_path(DIFFUSION, u, ep, e, t, dt))
        return _reconstructor(_us[1:], _es[:-1], _es[1:], Ts[1:], DTs[1:])
    
    return _to_path(us, es)


def calculate_functional_ess(sample_paths, normalise_by_chain=False):
    """
    Calculate MCMC ESS for scalar smoothing functionals.

    Parameters
    ----------
    sample_paths:        (n_samples, M, steps - 1, mesh_num, D) Reconstructed smoothing paths.
    normalise_by_chain:  If True, divides pooled multi-chain ESS by M.

    Returns
    -------
    ess_summary:     Median functional ESS.
    functional_ess:  ESS for each scalar smoothing functional.
    functionals:     Scalar functional traces.
    """
    n_samples, M = sample_paths.shape[:2]

    flat_paths = sample_paths.reshape(n_samples, M, -1, sample_paths.shape[-1])

    path_mean = jnp.mean(flat_paths, axis=(2, 3))
    path_energy = jnp.mean(flat_paths ** 2, axis=(2, 3))

    first_coord = flat_paths[..., 0]
    mid_coord = first_coord[:, :, first_coord.shape[2] // 2]
    terminal_coord = first_coord[:, :, -1]
    max_coord = jnp.max(first_coord, axis=2)
    roughness = jnp.mean(jnp.diff(first_coord, axis=2) ** 2, axis=2)

    functionals = jnp.stack(
        [
            path_mean,
            path_energy,
            mid_coord,
            terminal_coord,
            max_coord,
            roughness,
        ],
        axis=-1,
    )  # (n_samples, M, n_functionals)

    if M > 1:
        functional_ess = tfp.mcmc.effective_sample_size(functionals, filter_beyond_positive_pairs=True, cross_chain_dims=1)
        if normalise_by_chain:
            functional_ess = functional_ess / M

    else:
        functional_ess = tfp.mcmc.effective_sample_size(functionals[:, 0, :], filter_beyond_positive_pairs=True)

    ess_summary = jnp.median(functional_ess)
    return ess_summary, functional_ess, functionals


@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    data_key, init_key, adaptation_key, burnin_key, sample_key = jax.random.split(key, 5)

    true_xs, ys, *_ = get_data(data_key, PHI, SIGMA, OBS_SIGMA, args.D, DTs, args.mesh_num)

    kernel, init, adaptation_loop, experiment_loop = kernel_type.kernel_maker(
        ys, DRIFT, DIFFUSION, SIGMA, OBS_SIGMA, N=args.N,
        num=args.mesh_num, dts=DTs,
        resampling_func=resampling_fn,
        backward=args.backward,
        ancestor_move_func=last_step_fn,
        style=args.style, 
        conditional=True
    )
    adaptation_kernel = jax.jit(kernel)
    kernel = jax.jit(kernel)

    adaptation_loop = jax.jit(adaptation_loop, static_argnums=(2, 6), static_argnames=("window_size", "target_stat", "shared_delta", "shared_rho", "rho_direction"))
    experiment_loop = jax.jit(experiment_loop, static_argnums=(2, 3, 4, 5))

    csmc_kernel, csmc_init, *_ = get_csmc_kernel(
        ys, DRIFT, DIFFUSION, SIGMA, OBS_SIGMA, N=args.N,
        num=args.mesh_num, dts=DTs,
        resampling_func=resampling_fn,
        backward=True,
        ancestor_move_func=last_step_fn,
        style="guided", 
        conditional=False
    )

    init_xs, *_ = csmc_kernel(init_key, csmc_init(true_xs), None)
    init_state = init(init_xs)

    with jax.disable_jit(args.debug):
        adaptation_state, adapted_delta, adapted_rho, adaptation_hist, *_ = adaptation_loop(
            adaptation_key, init_state, adaptation_kernel,
            TARGET_ALPHA,
            DELTA, RHO, 
            args.adaptation,
            min_delta=MIN_DELTA, max_delta=MAX_DELTA,
            min_rho=MIN_RHO, max_rho=MAX_RHO,
            window_size=ADAPTATION_WINDOW,
            delta_rate=DELTA_ADAPTATION_RATE, delta_min_rate=DELTA_MIN_RATE,
            rho_rate=RHO_ADAPTATION_RATE, rho_min_rate=RHO_MIN_RATE,
            target_stat=TARGET_STAT,
            shared_delta=SHARED_DELTA, shared_rho=SHARED_RHO,
            rho_direction=RHO_DIRECTION
        )

    if args.verbose:
        jax.debug.print(
            "Adaptation delta median = {}, min = {}, max = {}", 
            jnp.median(adapted_delta), jnp.min(adapted_delta), jnp.max(adapted_delta)
        )
        jax.debug.print(
            "Adaptation rho median = {}, min = {}, max = {}", 
            jnp.median(adapted_rho), jnp.min(adapted_rho), jnp.max(adapted_rho)
        )

    _kernel = lambda k_, s: kernel(k_, s, adapted_delta, adapted_rho)
    burnin_keys = jax.random.split(burnin_key, args.M)
    sample_keys = jax.random.split(sample_key, args.M)

    def get_samples(sample_key_op, init_state_op, all_samples, n_samples):
        return experiment_loop(sample_key_op, init_state_op, _kernel, n_samples,
                               args.verbose, all_samples)

    with jax.disable_jit(args.debug):
        burnin_samples, burnin_pct = jax.vmap(get_samples, in_axes=[0, None, None, None], out_axes=0)(burnin_keys,
                                                                                                      adaptation_state,
                                                                                                      False,
                                                                                                      args.burnin)

    burnin_states = jax.vmap(init, in_axes=0)(burnin_samples)

    samples, ancestors, log_weights, pct = jax.vmap(get_samples, in_axes=[0, 0, None, None], out_axes=1)(sample_keys, 
                                                                                       burnin_states, 
                                                                                       True,
                                                                                       args.n_samples)

    sample_us, sample_es = samples
    sample_paths = to_path(sample_us, sample_es)
    _, ess_mcmc, *_ = calculate_functional_ess(sample_paths,normalise_by_chain=False)

    return ess_mcmc, adaptation_hist

# ESS storage
times_all = np.empty((args.K,))
ess_all = np.empty((args.K, 6))

# Adaptation storage
T_delta = 1 if SHARED_DELTA else args.steps
T_rho = 1 if SHARED_RHO else args.steps
delta_hist_all = np.empty((args.K, args.adaptation, T_delta))
rho_hist_all = np.empty((args.K, args.adaptation, T_rho))
delta_acc_rates_hist_all = np.empty((args.K, args.adaptation,))
rho_acc_rates_hist_all = np.empty((args.K, args.adaptation,))

# Warm up (remove JIT compilation time from runtime measurements)
# warmup_out = one_experiment(WARMUP_KEY)
# block_until_ready_tree(warmup_out)

# Compile once, without executing a full experiment
start = time.time()
compiled_one_experiment = one_experiment.lower(WARMUP_KEY).compile()
print(f"Compile time: {time.time() - start:.2f} seconds.")

for k, key_k in enumerate(tqdm.tqdm(EXPERIMENT_KEYS, desc="Experiment: ")):
    start = time.time()
    ess_k, adaptation_hist_k = compiled_one_experiment(key_k)
    block_until_ready_tree((ess_k, adaptation_hist_k))
    time_k = time.time() - start

    # ESS analysis
    times_all[k] = time_k
    ess_all[k] = ess_k

    # Adaptation analysis
    deltas_hist_k, rhos_hist_k, deltas_ar_hist_k, rhos_ar_hist_k = adaptation_hist_k
    delta_hist_all[k] = deltas_hist_k
    rho_hist_all[k] = rhos_hist_k
    delta_acc_rates_hist_all[k] = deltas_ar_hist_k
    rho_acc_rates_hist_all[k] = rhos_ar_hist_k
    
#     print(f"""
# Results:
#     - ESS-MCMC:   {ess_k}
#     - Time taken: {time_k}
# """)
#     print()

if not os.path.exists("results"):
    os.mkdir("results")

experiment_name = "kernel={},style={},D={},T={},N={},mesh-num={},steps={},M={},s={},a={},b={},seed={}"
experiment_name = experiment_name.format(
    kernel_type.name,
    args.style,
    args.D,
    args.T,
    args.N,
    args.mesh_num,
    args.steps,
    args.M,
    args.n_samples,
    args.adaptation,
    args.burnin,
    args.seed,
)

dirpath = f"results/{experiment_name}"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
# print(datapath)
np.savez_compressed(
    datapath,
    ess=ess_all,
    times=times_all,
    delta_hist=delta_hist_all,
    rho_hist=rho_hist_all,
    delta_acc_rates_hist=delta_acc_rates_hist_all,
    rho_acc_rates_hist=rho_acc_rates_hist_all,
)
