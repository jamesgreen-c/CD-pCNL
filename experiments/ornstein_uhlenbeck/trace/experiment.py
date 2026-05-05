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

# jax.config.update("jax_enable_x64", False)
# jax.config.update("jax_platform_name", "cpu")

# --- general config ---
MIN_DELTA = 1e-12
MAX_DELTA = 1e1
MIN_RHO = 1e-2
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
parser.set_defaults(verbose=True)

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
EXPERIMENT_KEYS = jax.random.split(KEY, args.K)

rho_m1 = 1 - args.rho
if args.rho_arg == "D":
    RHO = 1 - (rho_m1 / args.D ** args.rho_scale)
elif args.rho_arg == "T":
    RHO = 1 - (rho_m1 / args.T ** args.rho_scale)
elif args.rho_arg == "DT" or args.rho_arg == "TD":
    RHO = 1 - (rho_m1 / (args.D * args.T) ** args.rho_scale)
else:
    RHO = args.rho

if args.delta_arg == "D":
    DELTA = args.delta / args.D ** args.delta_scale
elif args.delta_arg == "T":
    DELTA = args.delta / args.T ** args.delta_scale
elif args.delta_arg == "DT" or args.delta_arg == "TD":
    DELTA = args.delta / (args.D * args.T) ** args.delta_scale
else:
    DELTA = args.delta


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


kernel_type = KernelType(args.kernel)
SHARED_DELTA = kernel_type.shared_delta()
SHARED_RHO = kernel_type.shared_rho()

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

    adaptation_loop = jax.jit(adaptation_loop, static_argnums=(2, 6), static_argnames=("window_size", "target_stat", "shared_delta", "shared_rho"))
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
        adaptation_state, adapted_delta, adapted_rho, *_ = adaptation_loop(
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
            shared_delta=SHARED_DELTA, shared_rho=SHARED_RHO
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

    # jax.debug.print("samples shape: {}", samples.shape)
    final_pct = jnp.mean(pct * 1.0, 0)
    final_pct = jnp.reshape(final_pct, (args.M, -1)) * jnp.ones((args.M, args.steps))
    
    # energy = log_pdf(samples, ys, m0, inv_chol_P0, F, b, inv_chol_Q)
    ess = jnp.exp(2.0 * logsumexp(log_weights, axis=-1) - logsumexp(2.0 * log_weights, axis=-1))   # (n_samples, M, steps)

    sample_us, sample_es = samples
    sample_paths = to_path(sample_us, sample_es)
    means_here = jnp.mean(sample_paths, 0)
    std_devs_here = jnp.std(sample_paths, 0)

    t_idx = jnp.array([0, args.steps // 2, args.steps - 2])
    traces = jnp.take(sample_paths[:, :, :, 0, 0], t_idx, axis=2)

    return (means_here, std_devs_here, final_pct, init_xs, true_xs, ys, adapted_delta, adapted_rho, traces, ess)

means_all = np.empty((args.K, args.M, args.steps - 1, args.mesh_num, args.D))
std_devs_all = np.empty((args.K, args.M, args.steps - 1, args.mesh_num, args.D))
final_pct_all = np.empty((args.K, args.M, args.steps))
traces_all = np.empty((args.K, args.n_samples, args.M, 3))
init_us_all = np.empty((args.K, args.steps, args.mesh_num + 1, args.D))
init_es_all = np.empty((args.K, args.steps, args.D))
ess_all = np.empty((args.K, args.n_samples, args.M, args.steps))

true_us_all = np.empty((args.K, args.steps, args.mesh_num + 1, args.D))
true_es_all = np.empty((args.K, args.steps, args.D))
ys_all = np.empty((args.K, args.steps, args.D))

for k, key_k in enumerate(tqdm.tqdm(EXPERIMENT_KEYS, desc="Experiment: ")):
    means_k, std_devs_k, final_pct_k, init_xs_k, true_xs_k, ys_k, delta_k, rho_k, traces_k, ess_k = one_experiment(key_k)
    true_us_k, true_es_k = true_xs_k
    init_us_k, init_es_k = init_xs_k

    # print(f"means shape: {means_k.shape}")
    # print(f"STD shape: {std_devs_k.shape}")
    # print(f"PCT shape: {final_pct_k.shape}")
    # print(f"Traces shape: {traces_k.shape}")

    # --- sample stuff ---
    means_all[k] = means_k
    std_devs_all[k] = std_devs_k
    final_pct_all[k] = final_pct_k
    traces_all[k] = traces_k
    init_us_all[k] = init_us_k
    init_es_all[k] = init_es_k
    ess_all[k] = ess_k

    # --- true stuff ---
    true_us_all[k] = true_us_k
    true_es_all[k] = true_es_k
    ys_all[k] = ys_k

    print(f"""
Results:
    - final min-max acceptance rate: {np.min(final_pct_k):.2%}, {np.max(final_pct_k):.2%}
    - final delta: {delta_k}
    - final rho: {rho_k}
    - final min-max ess: {np.min(ess_k):.2f}, {np.max(ess_k):.2f} 
    - final argmin-argmax ess: {np.argmin(ess_k)}, {np.argmax(ess_k)}
""")
    print()
    # - final min-max energy: {np.min(energy_k):.2E}, {np.max(energy_k):.2E}

if not os.path.exists("results"):
    os.mkdir("results")

experiment_name = "kernel={},style={},D={},N={},mesh-num={},steps={},M={},s={},a={},b={},seed={}"
experiment_name = experiment_name.format(
    kernel_type.name,
    args.style,
    args.D,
    args.N,
    args.mesh_num,
    args.steps,
    args.M,
    args.n_samples,
    args.adaptation,
    args.burnin,
    args.seed
)

dirpath = f"results/{experiment_name}"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
print(datapath)
np.savez_compressed(
    datapath, 
    means=means_all,
    std_devs=std_devs_all,
    final_pct=final_pct_all,
    traces=traces_all,
    true_us=true_us_all,
    true_es=true_es_all,
    ys=ys_all,
    init_us=init_us_all,
    init_es=init_es_all,
    ess=ess_all
)

