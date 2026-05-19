# ARGS PARSING
import argparse
import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

from itertools import product
from cd_ssm.utils.printing import ctext

from experiments.ornstein_uhlenbeck.kernels import KernelType

parser = argparse.ArgumentParser()
parser.add_argument("--i", dest="i", type=int, default=-1)
parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1
parser.add_argument("--M", dest="M", type=int, default=100)
parser.add_argument("--rho", dest="rho", type=float, default=.25)

parser.add_argument("--adaptation", dest="adaptation", type=int, default=500)
parser.add_argument("--burnin", dest="burnin", type=int, default=0)
parser.add_argument("--n-samples", dest="n_samples", type=int, default=1000)

args = parser.parse_args()


def results_exist(*, kernel, style, D, steps, mesh_num, args) -> bool:
    """ Mirror experiment.py's experiment_name + datapath convention and check if results already exist."""

    experiment_name = "kernel={},style={},D={},N={},mesh-num={},steps={},M={},s={},a={},b={},seed={}"
    experiment_name = experiment_name.format(
        kernel.name,
        style,
        D,
        args.N,
        mesh_num,
        steps,
        args.M,
        args.n_samples,
        args.adaptation,
        args.burnin,
        args.seed
    )

    datapath = os.path.join("results", experiment_name, "data.npz")
    return os.path.exists(datapath)


DS = (1, 10, 25, )
TS = (10,)
STEPS = (100, )
MESH_NUMS = (10, )

KERNELS = (
    KernelType.CSMC,
    KernelType.PCN,
    KernelType.RW_CSMC,
    KernelType.PCNL,
    KernelType.MALA_CSMC,
)

STYLES = (
    'guided',
    'na',
    'na',
    'na',
    'na',
)

combination = list(product(DS, TS, STEPS, MESH_NUMS, zip(KERNELS, STYLES)))
print(f"Number of experiments: {len(combination)}")

if args.i != -1 and not (0 <= args.i < len(combination)):
    raise ValueError(f"--i must be in [0, {len(combination)-1}] or -1, got {args.i}")

indices = range(len(combination)) if args.i == -1 else [args.i]

for j in indices:
    D, T, steps, mesh, (kernel, style, *_) = combination[j]

    if results_exist(kernel=kernel, style=style, D=D, steps=steps, mesh_num=mesh, args=args):
        print(ctext((f"Skipping (already run): kernel={kernel.name}, style={style}, ", 
                    f"T={T}, D={D}, steps={steps}, mesh-num={mesh}, ", 
                    f"N={args.N}, M={args.M}, "
                    f"n-samples={args.n_samples, }adaptation={args.adaptation}, burnin={args.burnin}"), "yellow"))
        continue

    exec_str = "python3 experiment.py --kernel {} --style {} --D {} --T {} --steps {} --mesh-num {} --N {} --M {} --n-samples {} --adaptation {} --burnin {}"
    exec_str = exec_str.format(kernel.value, style, D, T, steps, mesh, args.N, args.M, args.n_samples, args.adaptation, args.burnin)
    print("\nExecuting:", ctext(exec_str, "green"))
    # os.system(exec_str)
