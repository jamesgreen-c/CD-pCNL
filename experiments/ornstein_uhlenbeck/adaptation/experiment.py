import argparse
import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from experiments.ornstein_uhlenbeck.kernels import KernelType, get_csmc_kernel
from experiments.ornstein_uhlenbeck.model import get_data, get_dynamics
from cd_ssm.utils.common import force_move, barker_move
from cd_ssm.utils.kalman import sampling, filtering
from cd_ssm.utils.resamplings import killing, multinomial

# jax.config.update("jax_enable_x64", False)
# jax.config.update("jax_platform_name", "cpu")

# --- general config ---
MIN_DELTA = 1e-5
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
parser.add_argument("--M", dest="M", type=int, default=1)

parser.add_argument("--log-var", dest="log_var", type=float, default=0)
parser.add_argument("--phi", dest="phi", type=float, default=0.8)

parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--mesh-num", dest="mesh_num", type=int, default=50)

parser.add_argument("--adaptation", dest="adaptation", type=int, default=1000)
parser.add_argument("--rho-init", dest="rho_init", type=float, default=0.5)
parser.add_argument("--delta-init", dest="delta_init", type=float,
                    default=10 ** (0.5 * (np.log10(MIN_DELTA) + np.log10(MAX_DELTA))))

parser.add_argument("--target", dest="target", type=int, default=75)
parser.add_argument("--target-stat", dest='target_stat', type=str, default="mean")

parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.PCN)
parser.add_argument("--style", dest="style", type=str, default="na")

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
#############################################
#        LGSSM ADAPTATION EXPERIMENT        #
#############################################
Configuration:
    - T: {args.T}
    - kernel: {KernelType(args.kernel).name}
    - style: {args.style}
    - D: {args.D}
""")

# --- keys ---
KEY = jax.random.PRNGKey(args.seed)
EXPERIMENT_KEYS = jax.random.split(KEY, args.K)

# --- resampling config ---
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

if kernel_type.name == "RW_CSMC":
    # overwrite rho config
    RHO = args.delta_init / args.mesh_num
    MIN_RHO = MIN_DELTA / args.mesh_num
    MAX_RHO = MAX_DELTA
    RHO_MIN_RATE = DELTA_MIN_RATE
    RHO_ADAPTATION_RATE = DELTA_ADAPTATION_RATE
    RHO_DIRECTION = +1

else:
    RHO_DIRECTION = -1
    RHO = args.rho_init

# --- target acceptance rate ---
TARGET_ALPHA = args.target / 100 
if args.target_stat.isnumeric():
    TARGET_STAT = float(args.target_stat) / 100
else:
    TARGET_STAT = args.target_stat

print(f"""
ADAPTATION CONFIG:        
    - Target                     {TARGET_ALPHA}
    - delta init:                {args.delta_init}
    - min/max delta:             {MIN_DELTA}/{MAX_DELTA}
    - delta adaptation rate:     {DELTA_ADAPTATION_RATE}
    - rho init:                  {RHO}
    - min/max rho:               {MIN_RHO}/{MAX_RHO}
    - rho adaptation rate:       {RHO_ADAPTATION_RATE}
    - rho adaptation direction:  {RHO_DIRECTION}
""")


# --- dynamics config ---
SIGMA = 10 ** (args.log_var / 2)
PHI = args.phi
DTs = jnp.repeat(args.T / args.steps, args.steps)
DRIFT, DIFFUSION = get_dynamics(PHI, SIGMA)
OBS_SIGMA = 0.2


@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    data_key, init_key, adaptation_key, sample_key = jax.random.split(key, 4)

    true_xs, ys, *_ = get_data(data_key, PHI, SIGMA, OBS_SIGMA, args.D, DTs, args.mesh_num)

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

    kernel, init, adaptation_loop, *_ = kernel_type.kernel_maker(
        ys, DRIFT, DIFFUSION, SIGMA, OBS_SIGMA, N=args.N,
        num=args.mesh_num, dts=DTs,
        resampling_func=resampling_fn,
        backward=args.backward,
        ancestor_move_func=last_step_fn,
        style=args.style, 
        conditional=True
    )
    adaptation_kernel = jax.jit(kernel)
    init_state = init(init_xs)
    
    # --- adapt delta and rho
    adaptation_loop = jax.jit(adaptation_loop, static_argnums=(2, 6), static_argnames=("window_size", "target_stat", "shared_delta", "shared_rho", "rho_direction"))
    adaptation_state, adapted_delta, adapted_rho, adaptation_hist, *_ = adaptation_loop(
        adaptation_key, init_state, adaptation_kernel,
        TARGET_ALPHA,
        args.delta_init, RHO, 
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

    deltas_hist, rhos_hist, deltas_ar_hist, rhos_ar_hist = adaptation_hist
    return deltas_hist, rhos_hist, deltas_ar_hist, rhos_ar_hist


T_delta = 1 if SHARED_DELTA else args.steps
T_rho = 1 if SHARED_RHO else args.steps

delta_hist_all = np.empty((args.K, args.adaptation, T_delta))
rho_hist_all = np.empty((args.K, args.adaptation, T_rho))
delta_acc_rates_hist_all = np.empty((args.K, args.adaptation,))
rho_acc_rates_hist_all = np.empty((args.K, args.adaptation,))

for k, key_k in enumerate(tqdm.tqdm(EXPERIMENT_KEYS, desc="Experiment: ")):
    deltas_hist_k, rhos_hist_k, deltas_ar_hist_k, rhos_ar_hist_k = one_experiment(key_k)
    
    # print(f"deltas hist shape: {deltas_hist_k.shape}")
    # print(f"rhos hist shape: {rhos_hist_k.shape}")
    # print(f"deltas acceptance rate hist shape: {deltas_ar_hist_k.shape   }")
    # print(f"rhos acceptance rate hist shape: {rhos_ar_hist_k.shape}")

    delta_hist_all[k] = deltas_hist_k
    rho_hist_all[k] = rhos_hist_k
    delta_acc_rates_hist_all[k] = deltas_ar_hist_k
    rho_acc_rates_hist_all[k] = rhos_ar_hist_k

if not os.path.exists("results"):
    os.mkdir("results")

experiment_name = "kernel={},style={},adaptation={},target={},D={},N={},mesh-num={},steps={},seed={}"
experiment_name = experiment_name.format(
    kernel_type.name,
    args.style,
    args.adaptation,
    args.target,
    args.D,
    args.N,
    args.mesh_num,
    args.steps,
    args.seed
)

dirpath = f"results/{experiment_name}"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
np.savez_compressed(
    datapath, 
    delta_hist=delta_hist_all,
    rho_hist=rho_hist_all,
    delta_acc_rates_hist=delta_acc_rates_hist_all,
    rho_acc_rates_hist=rho_acc_rates_hist_all
)

