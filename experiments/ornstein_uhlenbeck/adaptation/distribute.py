# ARGS PARSING
import argparse
import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

from itertools import product
from cd_ssm.utils.printing import ctext

from experiments.ornstein_uhlenbeck.kernels import KernelType

parser = argparse.ArgumentParser()
parser.add_argument("--i", dest="i", type=int, default=-1)
parser.add_argument("--adaptation", dest="adaptation", default=1000)
parser.add_argument("--target", dest="target", type=int, default=27)
parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1
args = parser.parse_args()


def results_exist(*, kernel, style, D, steps, mesh_num, args) -> bool:
    """ Mirror experiment.py's experiment_name + datapath convention and check if results already exist."""

    experiment_name = (
        "kernel={},style={},adaptation={},target={},D={},N={},mesh-num={},steps={},seed={}"
    ).format(
        kernel.name,
        style,
        args.adaptation,
        args.target,
        D,
        args.N,
        mesh_num,
        steps,
        args.seed,
    )

    datapath = os.path.join("results", experiment_name, "data.npz")
    return os.path.exists(datapath)


# DS = (5, )
DS = (1, 5, 10, 20, 30, 40, 50, )# 75, 100, )

TS = (10,)

STEPS = (100,)
# STEPS = (10, 50, 100, 150, 200,)

# MESH_NUMS = (5, 10, 20, 40, 80, 160, 320, )
MESH_NUMS = (50,)

KERNELS = (
    KernelType.PCN,
    KernelType.RW_CSMC,
    KernelType.PCNL,
    KernelType.MALA_CSMC,
)

STYLES = (
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
        print(ctext(f"Skipping (already run): kernel={kernel.name}, style={style}, T={T}, D={D}, steps={steps}, mesh-num={mesh}, N={args.N}", "yellow"))
        continue

    exec_str = "python3 experiment.py --kernel {} --style {} --adaptation {} --target {} --D {} --T {} --steps {} --mesh-num {} --N {}"
    exec_str = exec_str.format(kernel.value, style, args.adaptation, args.target, D, T, steps, mesh, args.N)
    print("\nExecuting:", ctext(exec_str, "green"))
    # os.system(exec_str)
