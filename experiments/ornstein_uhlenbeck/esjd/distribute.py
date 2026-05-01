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
args = parser.parse_args()


def results_exist(*, kernel, style, rho, D, steps, mesh_num, args) -> bool:
    """ Mirror experiment.py's experiment_name + datapath convention and check if results already exist."""

    experiment_name = "kernel={},style={},rho={},D={},N={},mesh-num={},steps={},M={},seed={}"
    experiment_name = experiment_name.format(
        kernel.name,
        style,
        rho,
        D,
        args.N,
        mesh_num,
        steps,
        args.M,
        args.seed
    )

    datapath = os.path.join("results", experiment_name, "data.npz")
    return os.path.exists(datapath)


DS = (1, 5, 10, 25, 50, 75, 100, )
TS = (10,)
STEPS = (100, )
MESH_NUMS = (10, )

# STEPS = (10, 50, 100, 150, 200,)
# MESH_NUMS = (5, 10, 25, 50, 100, 200, )
# RHOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,)

KERNELS = (
    KernelType.CSMC,
    KernelType.PCN
)

STYLES = (
    'guided',
    'na',
)

combination = list(product(DS, TS, STEPS, MESH_NUMS, zip(KERNELS, STYLES)))
print(f"Number of experiments: {len(combination)}")

if args.i != -1 and not (0 <= args.i < len(combination)):
    raise ValueError(f"--i must be in [0, {len(combination)-1}] or -1, got {args.i}")

indices = range(len(combination)) if args.i == -1 else [args.i]

for j in indices:
    D, T, steps, mesh, rho, (kernel, style, *_) = combination[j]

    if results_exist(kernel=kernel, style=style, rho=rho, D=D, steps=steps, mesh_num=mesh, args=args):
        print(ctext(f"Skipping (already run): kernel={kernel.name}, style={style}, rho={rho}, T={T}, D={D}, steps={steps}, mesh-num={mesh}, N={args.N}, M={args.M}", "yellow"))
        continue

    exec_str = "python3 experiment.py --kernel {} --style {} --D {} --T {} --steps {} --mesh-num {} --rho {} --N {} --M {}"
    exec_str = exec_str.format(kernel.value, style, D, T, steps, mesh, rho, args.N, args.M)
    print("\nExecuting:", ctext(exec_str, "green"))
    # os.system(exec_str)
