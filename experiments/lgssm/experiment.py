import argparse
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from experiments.lgssm.kernels import KernelType, get_csmc_kernel
from experiments.lgssm.model import get_data, get_dynamics
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
parser.add_argument("--mesh-num", dest="mesh_num", type=int, default=100)

parser.add_argument("--delta", dest="delta", type=float,
                    default=1.)
parser.add_argument("--delta-scale", dest="delta_scale", type=float, default=1 / 3)
parser.add_argument("--delta-arg", dest="delta_arg", type=str, default="na")
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
SIGMA = 10 ** (args.log_var / 2)
PHI = args.phi
DTs = jnp.repeat(args.T / args.steps, args.steps)
# DTs = jnp.repeat(args.T / args.T, args.T)  # make dt = 1 so we can easily use kalman filter
DRIFT, DIFFUSION = get_dynamics(PHI, SIGMA)

def tic_fn(arr):
    time_elapsed = time.time() - NOW
    return np.array(time_elapsed, dtype=arr.dtype), arr


# @(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    data_key, init_key, sample_key = jax.random.split(key, 3)

    true_xs, ys, *_ = get_data(data_key, PHI, SIGMA, args.D, DTs, args.mesh_num)

    csmc_kernel, csmc_init, _ = get_csmc_kernel(
        ys, DRIFT, DIFFUSION, SIGMA, N=args.N,
        num=args.mesh_num, dts=DTs,
        resampling_func=resampling_fn,
        backward=args.backward,
        ancestor_move_func=last_step_fn,
        style=args.style, 
        conditional=False
    )

    init_xs, *_ = csmc_kernel(init_key, csmc_init(true_xs), None)
    init_state = init(init_xs)

    kernel_, init, _ = kernel_type.kernel_maker(
        ys, DRIFT, DIFFUSION, SIGMA, N=args.N,
        num=args.mesh_num, dts=DTs,
        resampling_func=resampling_fn,
        backward=args.backward,
        ancestor_move_func=last_step_fn,
        style=args.style, 
        conditional=True
    )
    kernel_ = jax.jit(kernel_)

    def esjd(k_):
        xs, *_ = init_state
        next_xs, *_ = kernel_(k_, init_state, DELTA)
        return jnp.sum((xs - next_xs) ** 2, -1)

    sample_keys = jax.random.split(sample_key, args.M)
    esjd_vals = jax.vmap(esjd)(sample_keys)
    return esjd_vals.mean(0)


esjd_all = np.empty((args.K, args.T))
for k, key_k in enumerate(tqdm.tqdm(EXPERIMENT_KEYS, desc="Experiment: ")):
    esjd_k = one_experiment(key_k)
    esjd_all[k] = esjd_k

if not os.path.exists("results"):
    os.mkdir("results")

experiment_name = "D={},N={},mesh-num={},steps={},seed={}"
experiment_name = experiment_name.format(
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
    esjd=esjd_all,
)
