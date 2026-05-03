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

from cd_ssm import bridge
from cd_ssm.utils.common import force_move, barker_move
from cd_ssm.utils.kalman import sampling, filtering
from cd_ssm.utils.resamplings import killing, multinomial

# jax.config.update("jax_enable_x64", False)
# jax.config.update("jax_platform_name", "cpu")

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

parser.add_argument("--rho", dest="rho", type=float, default=.25)
parser.add_argument("--rho-scale", dest="rho_scale", type=float, default=1/5)
parser.add_argument("--rho-arg", dest="rho_arg", type=str, default="D")

parser.add_argument("--delta", dest="delta", type=float, default=1)
parser.add_argument("--delta-scale", dest="delta_scale", type=float, default=1)
parser.add_argument("--delta-arg", dest="delta_arg", type=str, default="D")

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
    - T: {args.T}
    - kernel: {KernelType(args.kernel).name}
    - style: {args.style}
    - D: {args.D}
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
DELTA = kernel_type.shape_delta(DELTA, args.steps)

SIGMA = 10 ** (args.log_var / 2)
PHI = args.phi
DTs = jnp.repeat(args.T / args.steps, args.steps)
# DTs = jnp.repeat(args.T / args.T, args.T)  # make dt = 1 so we can easily use kalman filter
Ts = jnp.cumsum(DTs)
DRIFT, DIFFUSION = get_dynamics(PHI, SIGMA)
OBS_SIGMA = 0.2

def tic_fn(arr):
    time_elapsed = time.time() - NOW
    return np.array(time_elapsed, dtype=arr.dtype), arr


@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    data_key, init_key, sample_key = jax.random.split(key, 3)

    true_xs, ys, *_ = get_data(data_key, PHI, SIGMA, OBS_SIGMA, args.D, DTs, args.mesh_num, )

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

    kernel_, init, *_ = kernel_type.kernel_maker(
        ys, DRIFT, DIFFUSION, SIGMA, OBS_SIGMA, N=args.N,
        num=args.mesh_num, dts=DTs,
        resampling_func=resampling_fn,
        backward=args.backward,
        ancestor_move_func=last_step_fn,
        style=args.style, 
        conditional=True
    )

    kernel_ = jax.jit(kernel_)
    init_state = init(init_xs)

    to_path = jax.vmap(lambda u, ep, e, t, dt: bridge.to_path(DIFFUSION, u, ep, e, t, dt))
    def esjd(k_):
        xs, *_ = init_state
        next_xs, *_ = kernel_(k_, init_state, DELTA, RHO)
        us, es = xs
        next_us, next_es = next_xs

        # --- map sample back to path space ---
        path = to_path(us[1:], es[:-1], es[1:], Ts[1:], DTs[1:])  # (T - 1, M, D)
        next_path = to_path(next_us[1:], next_es[:-1], next_es[1:], Ts[1:], DTs[1:])  # (T - 1, M, D)
        
        # --- calculate esjd ---
        _esjd = jnp.sum((next_path - path) ** 2, axis=(1, 2))
        return _esjd, next_xs, next_path

    sample_keys = jax.random.split(sample_key, args.M)

    (esjd_vals, samples, sample_paths) = jax.vmap(esjd)(sample_keys)
    return esjd_vals.mean(0), samples, sample_paths, true_xs, ys, init_xs


us_all = np.empty((args.K, args.M, args.steps, args.mesh_num + 1, args.D))
es_all = np.empty((args.K, args.M, args.steps, args.D))
true_us_all = np.empty((args.K, args.steps, args.mesh_num + 1, args.D))
true_es_all = np.empty((args.K, args.steps, args.D))
ys_all = np.empty((args.K, args.steps, args.D))
init_us_all = np.empty((args.K, args.steps, args.mesh_num + 1, args.D))
init_es_all = np.empty((args.K, args.steps, args.D))
esjd_all = np.empty((args.K, args.steps - 1))
paths_all = np.empty((args.K, args.M, args.steps - 1, args.mesh_num, args.D))

for k, key_k in enumerate(tqdm.tqdm(EXPERIMENT_KEYS, desc="Experiment: ")):
    esjd_k, samples_k, sample_paths_k, true_xs_k, ys_k, init_xs_k = one_experiment(key_k)
    us_k, es_k = samples_k
    true_us_k, true_es_k = true_xs_k
    init_us_k, init_es_k = init_xs_k

    # print(f"True us shape: {true_us_k.shape}")
    # print(f"True es shape: {true_es_k.shape}")
    # print(f"Ys shape: {ys_k.shape}")
    # print(f"Init us shape: {init_us_k.shape}")
    # print(f"Init es shape: {init_es_k.shape}")
    # print(sample_paths_k.shape)
    # print(esjd_k.shape)

    us_all[k] = us_k
    es_all[k] = es_k
    true_us_all[k] = true_us_k
    true_es_all[k] = true_es_k
    ys_all[k] = ys_k
    init_us_all[k] = init_us_k
    init_es_all[k] = init_es_k
    esjd_all[k] = esjd_k
    paths_all[k] = sample_paths_k

if not os.path.exists("results"):
    os.mkdir("results")

experiment_name = "kernel={},style={},rho={},D={},N={},mesh-num={},steps={},M={},seed={}"
experiment_name = experiment_name.format(
    kernel_type.name,
    args.style,
    RHO,
    args.D,
    args.N,
    args.mesh_num,
    args.steps,
    args.M,
    args.seed
)

dirpath = f"results/{experiment_name}"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
print(datapath)
np.savez_compressed(
    datapath, 
    esjd=esjd_all,
    paths=paths_all,
    us=us_all,
    es=es_all,
    true_us=true_us_all,
    true_es=true_es_all,
    ys=ys_all,
    init_us=init_us_all,
    init_es=init_es_all
)

