import argparse
import os
from itertools import product

from cd_ssm.utils.printing import ctext
from experiments.ornstein_uhlenbeck.kernels import KernelType


parser = argparse.ArgumentParser()

parser.add_argument("--i", dest="i", type=int, default=-1)
parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--N", dest="N", type=int, default=31)
parser.add_argument("--M", dest="M", type=int, default=100)

parser.add_argument("--rho", dest="rho", type=float, default=.25)

parser.add_argument("--adaptation", dest="adaptation", type=int, default=3000)
parser.add_argument("--burnin", dest="burnin", type=int, default=1000)
parser.add_argument("--n-samples", dest="n_samples", type=int, default=2000)

parser.add_argument(
    "--mode",
    dest="mode",
    type=str,
    default="all",
    choices=("dmesh", "T", "steps", "all"),
)

parser.add_argument("--run", dest="run", action="store_true")
parser.add_argument("--no-run", dest="run", action="store_false")
parser.set_defaults(run=False)

args = parser.parse_args()


# ---------------------------------------------------------------------
# Baseline controls
# ---------------------------------------------------------------------
BASE_D = 10
BASE_T = 10
BASE_STEPS = 100
BASE_MESH_NUM = 25

# Main heatmap grid.
DS = (1, 5, 10, 25, 50, 75, 100)
MESH_NUMS = (5, 10, 25, 50, 75, 100)

# ESS_MCMC sweeps.
TS = (2, 5, 10, 20, 50, 100)
STEPS = (10, 25, 50, 75, 100, 150, 200)

KERNELS = (
    (KernelType.CSMC, "guided"),
    (KernelType.PCN, "na"),
    (KernelType.PCNL, "na"),
    (KernelType.RW_CSMC, "na"),
    (KernelType.MALA_CSMC, "na"),
)


def experiment_id(*, suite, D, T, steps, mesh_num, kernel, style):
    return {
        "suite": suite,
        "D": int(D),
        "T": int(T),
        "steps": int(steps),
        "mesh_num": int(mesh_num),
        "kernel": kernel,
        "style": style,
    }


def results_exist(*, kernel, style, D, T, steps, mesh_num, args) -> bool:
    """
    Mirror experiment.py's experiment_name + datapath convention.
    Must match experiment.py exactly.
    """
    experiment_name = "kernel={},style={},D={},T={},N={},mesh-num={},steps={},M={},s={},a={},b={},seed={}"
    experiment_name = experiment_name.format(
        kernel.name,
        style,
        D,
        T,
        args.N,
        mesh_num,
        steps,
        args.M,
        args.n_samples,
        args.adaptation,
        args.burnin,
        args.seed,
    )

    datapath = os.path.join("results", experiment_name, "data.npz")
    return os.path.exists(datapath)


def build_combinations(mode):
    combinations = []

    if mode in ("dmesh", "all"):
        # Heatmaps:
        #   (D, mesh_num) -> delta*, rho*, replacement rate, ESS_MCMC
        # Fixed: T, steps
        for D, mesh_num, (kernel, style) in product(DS, MESH_NUMS, KERNELS):
            combinations.append(
                experiment_id(
                    suite="dmesh",
                    D=D,
                    T=BASE_T,
                    steps=BASE_STEPS,
                    mesh_num=mesh_num,
                    kernel=kernel,
                    style=style,
                )
            )

    if mode in ("T", "all"):
        # ESS_MCMC vs T.
        # Fixed: D, mesh_num, steps
        for T, (kernel, style) in product(TS, KERNELS):
            combinations.append(
                experiment_id(
                    suite="T",
                    D=BASE_D,
                    T=T,
                    steps=BASE_STEPS,
                    mesh_num=BASE_MESH_NUM,
                    kernel=kernel,
                    style=style,
                )
            )

    if mode in ("steps", "all"):
        # ESS_MCMC vs steps.
        # Fixed: D, T, mesh_num
        for steps, (kernel, style) in product(STEPS, KERNELS):
            combinations.append(
                experiment_id(
                    suite="steps",
                    D=BASE_D,
                    T=BASE_T,
                    steps=steps,
                    mesh_num=BASE_MESH_NUM,
                    kernel=kernel,
                    style=style,
                )
            )

    return combinations


COMBINATIONS = build_combinations(args.mode)

print(f"Mode:                  {args.mode}")
print(f"Number of experiments: {len(COMBINATIONS)}")
print(f"Base D:                {BASE_D}")
print(f"Base T:                {BASE_T}")
print(f"Base steps:            {BASE_STEPS}")
print(f"Base mesh_num:         {BASE_MESH_NUM}")
print(f"D grid:                {DS}")
print(f"mesh grid:             {MESH_NUMS}")
print(f"T sweep:               {TS}")
print(f"steps sweep:           {STEPS}")
print(f"kernels:               {[k.name for k, _ in KERNELS]}")

if args.i != -1 and not (0 <= args.i < len(COMBINATIONS)):
    raise ValueError(f"--i must be in [0, {len(COMBINATIONS) - 1}] or -1, got {args.i}")

indices = range(len(COMBINATIONS)) if args.i == -1 else [args.i]


for j in indices:
    combo = COMBINATIONS[j]

    suite = combo["suite"]
    D = combo["D"]
    T = combo["T"]
    steps = combo["steps"]
    mesh_num = combo["mesh_num"]
    kernel = combo["kernel"]
    style = combo["style"]

    if results_exist(kernel=kernel, style=style, D=D, T=T, steps=steps, mesh_num=mesh_num, args=args):
        print(
            ctext(
                (
                    f"Skipping already run: "
                    f"suite={suite}, kernel={kernel.name}, style={style}, "
                    f"T={T}, D={D}, steps={steps}, mesh-num={mesh_num}, "
                    f"N={args.N}, M={args.M}, "
                    f"n-samples={args.n_samples}, adaptation={args.adaptation}, burnin={args.burnin}"
                ),
                "yellow",
            )
        )
        continue

    exec_str = (
        "python3 experiment.py "
        "--kernel {} "
        "--style {} "
        "--D {} "
        "--T {} "
        "--steps {} "
        "--mesh-num {} "
        "--N {} "
        "--M {} "
        "--rho {} "
        "--n-samples {} "
        "--adaptation {} "
        "--burnin {} "
        "--seed {}"
    )

    exec_str = exec_str.format(
        kernel.value,
        style,
        D,
        T,
        steps,
        mesh_num,
        args.N,
        args.M,
        args.rho,
        args.n_samples,
        args.adaptation,
        args.burnin,
        args.seed,
    )

    print(
        "\nExecuting:",
        ctext(
            (
                f"[{j}/{len(COMBINATIONS) - 1}] "
                f"suite={suite} :: {exec_str}"
            ),
            "green",
        ),
    )

    if args.run:
        os.system(exec_str)