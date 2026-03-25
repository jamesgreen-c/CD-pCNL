import argparse
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from experiments.lgssm.kernels import KernelType, get_filter_csmc_kernel
from experiments.lgssm.model import get_data, get_dynamics
from cd_ssm.utils.common import force_move, barker_move
from cd_ssm.utils.kalman import sampling, filtering
from cd_ssm.utils.resamplings import killing, multinomial, dynamic

# jax.config.update("jax_enable_x64", False)
# jax.config.update("jax_platform_name", "cpu")

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=10)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--K", dest="K", type=int, default=3)
parser.add_argument("--M", dest="M", type=int, default=1)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)
parser.add_argument("--phi", dest="phi", type=float, default=0.8)

parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--mesh-num", dest="mesh_num", type=int, default=3)

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

parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--threshold", type=float, default=0.5)
parser.set_defaults(dynamic=False)

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
    resampling_func = killing
elif args.resampling == "multinomial":
    resampling_func = multinomial
else:
    raise ValueError(f"Unknown resampling {args.resampling}")

if args.dynamic:
    assert args.threshold is not None, "If using dynamic sampling, please provide a threshold for the ESS"
    def resampling_fn(key, weights, i, j, conditional):
        return dynamic(resampling_func, args.threshold, key, weights, i, j, conditional)
else:
    resampling_fn = resampling_func

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


@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    data_key, init_key, sample_key = jax.random.split(key, 3)

    true_xs, ys, As, chol_Qs, chol_P0 = get_data(data_key, PHI, SIGMA, args.D, DTs)

    kernel_, init, = get_filter_csmc_kernel(
        ys, DRIFT, DIFFUSION, SIGMA, N=args.N,
        num=args.mesh_num-1, dts=DTs,
        resampling_func=resampling_fn,
        # backward=args.backward,
        # ancestor_move_func=last_step_fn,
        style=args.style, 
        conditional=False
    )
    # kernel_ = jax.jit(kernel_)

    As, _, _, _, _, xs = kernel_(sample_key, init(true_xs), None)
    return As, true_xs, xs, ys



true_xs_all = np.empty((args.K, args.steps, args.D))
xs_all = np.empty((args.K, args.steps, args.N+1, args.mesh_num, args.D))
ys_all = np.empty((args.K, args.steps, args.D))
As_all = np.empty((args.K, args.steps-1, args.N+1,))

for k, key_k in enumerate(tqdm.tqdm(EXPERIMENT_KEYS, desc="Experiment: ")):
    As_k, true_xs_k, xs_k, ys_k = one_experiment(key_k)

    true_xs_all[k, ...] = true_xs_k
    ys_all[k, ...] = ys_k
    xs_all[k, ...] = xs_k
    As_all[k, ...] = As_k

if not os.path.exists("filter-results"):
    os.mkdir("filter-results")

experiment_name = "D={},N={},mesh-num={},steps{},seed={}"
experiment_name = experiment_name.format(
    args.D,
    args.N,
    args.mesh_num,
    args.steps,
    args.seed
)

dirpath = f"filter-results/{experiment_name}"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
np.savez_compressed(
    datapath, 
    true_xs=true_xs_all,
    ys=ys_all,
    As=As_all,
    xs=xs_all
)