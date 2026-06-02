"""
PLOT ESS_MCMC AND FINAL ADAPTED PARAMETERS

This script assumes the experiment runner saves files with the convention

    results/kernel={KERNEL},style={STYLE},D={D},T={T},N={N},mesh-num={MESH},steps={STEPS},M={M},s={N_SAMPLES},a={ADAPTATION},b={BURNIN},seed={SEED}/data.npz

and that each data.npz contains

    ess                       (K, 6)
    times                     (K,)
    delta_hist                (K, adaptation, T_delta)
    rho_hist                  (K, adaptation, T_rho)
    delta_acc_rates_hist       (K, adaptation)
    rho_acc_rates_hist         (K, adaptation)

Outputs
-------
1. Four line plots:
       ESS_MCMC vs D
       ESS_MCMC vs mesh_num
       ESS_MCMC vs T
       ESS_MCMC vs steps

2. For every kernel/style pair, one heatmap figure over (D, mesh_num):
       final adapted delta
       final adapted rho

Run from the same directory as your results/ folder, for example:

    python3 plot_ess_mcmc_summary.py

or

    python3 plot_ess_mcmc_summary.py --results-dir results --out-dir results/summary_plots
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from experiments.ornstein_uhlenbeck.kernels import KernelType


parser = argparse.ArgumentParser()

parser.add_argument("--results-dir", type=str, default="results")
parser.add_argument("--out-dir", type=str, default="results/summary_plots")

parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--N", type=int, default=31)
parser.add_argument("--M", type=int, default=4)
parser.add_argument("--adaptation", type=int, default=2_000)
parser.add_argument("--burnin", type=int, default=500)
parser.add_argument("--n-samples", dest="n_samples", type=int, default=1_000)

parser.add_argument(
    "--summary-stat",
    type=str,
    default="median",
    choices=("median", "mean"),
    help="How to collapse ESS over independent experiment repeats and the six scalar functionals.",
)
parser.add_argument(
    "--param-stat",
    type=str,
    default="median",
    choices=("median", "mean"),
    help="How to collapse final delta/rho over repeats and time coordinates when not shared.",
)
parser.add_argument(
    "--yscale",
    type=str,
    default="linear",
    choices=("linear", "log"),
    help="Y-axis scale for ESS line plots.",
)
parser.add_argument(
    "--show-missing",
    action="store_true",
    help="Print every missing data.npz path.",
)
parser.add_argument(
    "--ess-functional",
    type=str,
    default="path_mean",
    choices=(
        "path_mean",
        "path_energy",
        "mid_coord",
        "terminal_coord",
        "max_coord",
        "roughness",
        "all",
    ),
    help=(
        "Which scalar smoothing functional to use for ESS_MCMC. "
        "Use 'all' to recover the old behaviour of summarising over all six functionals."
    ),
)

args = parser.parse_args()

# ---------------------------------------------------------------------
# Experiment grid: mirrors the experiment-launch file.
# ---------------------------------------------------------------------
BASE_D = 1
BASE_T = 10
BASE_STEPS = 100
BASE_MESH_NUM = 25

DS = (1, 5, 10, 25, 50, 75, 100)
MESH_NUMS = (5, 10, 25, 50, 75, 100)
TS = (2, 5, 10, 20, 50, 100)
STEPS = (10, 25, 50, 75, 100, 150, 200)

KERNELS = (
    (KernelType.CSMC, "guided"),
    (KernelType.PCN, "na"),
    (KernelType.PCNL, "na"),
    (KernelType.RW_CSMC, "na"),
    (KernelType.MALA_CSMC, "na"),
)

FUNCTIONAL_NAMES = (
    "path_mean",
    "path_energy",
    "mid_coord",
    "terminal_coord",
    "max_coord",
    "roughness",
)

FUNCTIONAL_TO_INDEX = {name: i for i, name in enumerate(FUNCTIONAL_NAMES)}

@dataclass(frozen=True)
class ResultSummary:
    ess: float
    ess_q25: float
    ess_q75: float
    delta: float
    rho: float
    time: float


def select_ess_values(ess_all: np.ndarray, ess_functional: str) -> np.ndarray:
    """
    Select ESS values for one scalar smoothing functional.

    Parameters
    ----------
    ess_all:          ESS array with shape (K, 6), where K is the number of
                      independent experiment repeats.
    ess_functional:   One of FUNCTIONAL_NAMES, or "all".

    Returns
    -------
    ess_values:       If ess_functional is named, shape (K,). If "all", shape
                      (K * 6,), recovering the old behaviour.
    """
    ess_all = np.asarray(ess_all, dtype=float)

    if ess_functional == "all":
        return ess_all.reshape(-1)

    functional_idx = FUNCTIONAL_TO_INDEX[ess_functional]
    return ess_all[:, functional_idx]


def experiment_name(*, kernel, style, D, T, steps, mesh_num, args) -> str:
    """Mirror experiment.py exactly."""
    template = "kernel={},style={},D={},T={},N={},mesh-num={},steps={},M={},s={},a={},b={},seed={}"
    return template.format(
        kernel.name,
        style,
        int(D),
        int(T),
        args.N,
        int(mesh_num),
        int(steps),
        args.M,
        args.n_samples,
        args.adaptation,
        args.burnin,
        args.seed,
    )


def datapath(*, kernel, style, D, T, steps, mesh_num, args) -> str:
    return os.path.join(
        args.results_dir,
        experiment_name(
            kernel=kernel,
            style=style,
            D=D,
            T=T,
            steps=steps,
            mesh_num=mesh_num,
            args=args,
        ),
        "data.npz",
    )


def collapse(x, stat: str) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    if stat == "mean":
        return float(np.nanmean(x))
    return float(np.nanmedian(x))


def summarise_result(path: str, args) -> Optional[ResultSummary]:
    if not os.path.exists(path):
        if args.show_missing:
            print(f"Missing: {path}")
        return None

    with np.load(path) as data:
        ess_all = np.asarray(data["ess"], dtype=float)           # (K, 6)
        times = np.asarray(data["times"], dtype=float)           # (K,)
        delta_hist = np.asarray(data["delta_hist"], dtype=float) # (K, adaptation, T_delta)
        rho_hist = np.asarray(data["rho_hist"], dtype=float)     # (K, adaptation, T_rho)

    # ESS_MCMC is stored per repeat and per scalar smoothing functional.
    ess_values = select_ess_values(ess_all, args.ess_functional)
    ess = collapse(ess_values, args.summary_stat)
    ess_q25 = float(np.nanpercentile(ess_values, 25)) if not np.all(np.isnan(ess_values)) else np.nan
    ess_q75 = float(np.nanpercentile(ess_values, 75)) if not np.all(np.isnan(ess_values)) else np.nan

    # Final adapted parameters. If delta/rho are not shared, this also collapses over
    # the time-coordinate dimension, which gives one scalar per experiment setting.
    final_delta = delta_hist[:, -1, :]
    final_rho = rho_hist[:, -1, :]

    return ResultSummary(
        ess=ess,
        ess_q25=ess_q25,
        ess_q75=ess_q75,
        delta=collapse(final_delta, args.param_stat),
        rho=collapse(final_rho, args.param_stat),
        time=collapse(times, args.summary_stat),
    )


def load_summary(*, kernel, style, D, T, steps, mesh_num, args) -> Optional[ResultSummary]:
    path = datapath(
        kernel=kernel,
        style=style,
        D=D,
        T=T,
        steps=steps,
        mesh_num=mesh_num,
        args=args,
    )
    return summarise_result(path, args)


def label(kernel, style: str) -> str:
    return f"{kernel.name}-{style}"

def maybe_set_log_y(ax, yscale: str):
    if yscale == "log":
        ax.set_yscale("log")


def plot_ess_sweep(*, xs, xname, values_by_kernel, outpath, args):
    fig, ax = plt.subplots(figsize=(10, 6))

    for kernel, style in KERNELS:
        ys = values_by_kernel[(kernel.name, style)]
        ax.plot(xs, np.log(ys), marker="o", label=label(kernel, style))

    ax.set_xlabel(xname)
    ax.set_ylabel("Log ESS_MCMC")
    ax.set_title(f"ESS_MCMC vs {xname}")
    maybe_set_log_y(ax, args.yscale)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def collect_ess_vs_D(args):
    values = {}
    for kernel, style in KERNELS:
        ys = []
        for D in DS:
            result = load_summary(
                kernel=kernel,
                style=style,
                D=D,
                T=BASE_T,
                steps=BASE_STEPS,
                mesh_num=BASE_MESH_NUM,
                args=args,
            )
            ys.append(np.nan if result is None else result.ess)
        values[(kernel.name, style)] = np.asarray(ys, dtype=float)
    return values


def collect_ess_vs_mesh_num(args):
    values = {}
    for kernel, style in KERNELS:
        ys = []
        for mesh_num in MESH_NUMS:
            result = load_summary(
                kernel=kernel,
                style=style,
                D=BASE_D,
                T=BASE_T,
                steps=BASE_STEPS,
                mesh_num=mesh_num,
                args=args,
            )
            ys.append(np.nan if result is None else result.ess)
        values[(kernel.name, style)] = np.asarray(ys, dtype=float)
    return values


def collect_ess_vs_T(args):
    values = {}
    for kernel, style in KERNELS:
        ys = []
        for T in TS:
            result = load_summary(
                kernel=kernel,
                style=style,
                D=BASE_D,
                T=T,
                steps=BASE_STEPS,
                mesh_num=BASE_MESH_NUM,
                args=args,
            )
            ys.append(np.nan if result is None else result.ess)
        values[(kernel.name, style)] = np.asarray(ys, dtype=float)
    return values


def collect_ess_vs_steps(args):
    values = {}
    for kernel, style in KERNELS:
        ys = []
        for steps in STEPS:
            result = load_summary(
                kernel=kernel,
                style=style,
                D=BASE_D,
                T=BASE_T,
                steps=steps,
                mesh_num=BASE_MESH_NUM,
                args=args,
            )
            ys.append(np.nan if result is None else result.ess)
        values[(kernel.name, style)] = np.asarray(ys, dtype=float)
    return values


def collect_dmesh_param_grids(*, kernel, style, args):
    delta_grid = np.full((len(DS), len(MESH_NUMS)), np.nan)
    rho_grid = np.full((len(DS), len(MESH_NUMS)), np.nan)
    ess_grid = np.full((len(DS), len(MESH_NUMS)), np.nan)

    for i, D in enumerate(DS):
        for j, mesh_num in enumerate(MESH_NUMS):
            result = load_summary(
                kernel=kernel,
                style=style,
                D=D,
                T=BASE_T,
                steps=BASE_STEPS,
                mesh_num=mesh_num,
                args=args,
            )
            if result is None:
                continue
            delta_grid[i, j] = result.delta
            rho_grid[i, j] = result.rho
            ess_grid[i, j] = result.ess

    return delta_grid, rho_grid, ess_grid


def add_heatmap(ax, grid, *, title, cbar_label, vmin=None, vmax=None, log_values=False):
    plot_grid = np.asarray(grid, dtype=float)
    cbar_label_ = cbar_label

    if log_values:
        plot_grid = np.where(plot_grid > 0, np.log(plot_grid), np.nan)
        cbar_label_ = f"log {cbar_label}"

    masked = np.ma.masked_invalid(plot_grid)
    im = ax.imshow(masked, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("mesh_num")
    ax.set_ylabel("D")
    ax.set_xticks(np.arange(len(MESH_NUMS)))
    ax.set_xticklabels(MESH_NUMS)
    ax.set_yticks(np.arange(len(DS)))
    ax.set_yticklabels(DS)

    for i in range(len(DS)):
        for j in range(len(MESH_NUMS)):
            val = plot_grid[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=7)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label_)


def plot_param_heatmaps(args):
    for kernel, style in KERNELS:
        delta_grid, rho_grid, _ = collect_dmesh_param_grids(kernel=kernel, style=style, args=args)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Final adapted parameters: {label(kernel, style)}")

        add_heatmap(
            ax[0],
            delta_grid,
            title="Final adapted delta",
            cbar_label="delta",
            log_values=False,
        )
        add_heatmap(
            ax[1],
            rho_grid,
            title="Final adapted rho",
            cbar_label="rho",
            log_values=False,
        )

        fig.tight_layout()
        outpath = os.path.join(args.out_dir, f"heatmap_delta_rho_{kernel.name}_{style}.png")
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Log-scale versions are often more readable for tuning parameters.
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Final adapted parameters, log scale: {label(kernel, style)}")

        add_heatmap(
            ax[0],
            delta_grid,
            title="Final adapted delta",
            cbar_label="delta",
            log_values=True,
        )
        add_heatmap(
            ax[1],
            rho_grid,
            title="Final adapted rho",
            cbar_label="rho",
            log_values=True,
        )

        fig.tight_layout()
        outpath = os.path.join(args.out_dir, f"heatmap_log_delta_log_rho_{kernel.name}_{style}.png")
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)


# def write_summary_csv(args):
#     outpath = os.path.join(args.out_dir, "summary_table.csv")
#     rows = [
#         "suite,kernel,style,D,T,steps,mesh_num,ess,ess_q25,ess_q75,delta,rho,time,path"
#     ]

#     def add_row(suite, kernel, style, D, T, steps, mesh_num):
#         path = datapath(
#             kernel=kernel,
#             style=style,
#             D=D,
#             T=T,
#             steps=steps,
#             mesh_num=mesh_num,
#             args=args,
#         )
#         result = summarise_result(path, args)
#         if result is None:
#             return
#         rows.append(
#             ",".join(
#                 [
#                     suite,
#                     kernel.name,
#                     style,
#                     str(D),
#                     str(T),
#                     str(steps),
#                     str(mesh_num),
#                     f"{result.ess:.10g}",
#                     f"{result.ess_q25:.10g}",
#                     f"{result.ess_q75:.10g}",
#                     f"{result.delta:.10g}",
#                     f"{result.rho:.10g}",
#                     f"{result.time:.10g}",
#                     path,
#                 ]
#             )
#         )

#     for kernel, style in KERNELS:
#         for D in DS:
#             add_row("D", kernel, style, D, BASE_T, BASE_STEPS, BASE_MESH_NUM)
#         for mesh_num in MESH_NUMS:
#             add_row("mesh_num", kernel, style, BASE_D, BASE_T, BASE_STEPS, mesh_num)
#         for T in TS:
#             add_row("T", kernel, style, BASE_D, T, BASE_STEPS, BASE_MESH_NUM)
#         for steps in STEPS:
#             add_row("steps", kernel, style, BASE_D, BASE_T, steps, BASE_MESH_NUM)

#     with open(outpath, "w", encoding="utf-8") as f:
#         f.write("\n".join(rows) + "\n")


def main():
    os.makedirs(args.out_dir, exist_ok=True)

    ess_vs_D = collect_ess_vs_D(args)
    ess_vs_mesh_num = collect_ess_vs_mesh_num(args)
    ess_vs_T = collect_ess_vs_T(args)
    ess_vs_steps = collect_ess_vs_steps(args)

    plot_ess_sweep(
        xs=DS,
        xname="D",
        values_by_kernel=ess_vs_D,
        outpath=os.path.join(args.out_dir, "ess_mcmc_vs_D.png"),
        args=args,
    )
    plot_ess_sweep(
        xs=MESH_NUMS,
        xname="mesh_num",
        values_by_kernel=ess_vs_mesh_num,
        outpath=os.path.join(args.out_dir, "ess_mcmc_vs_mesh_num.png"),
        args=args,
    )
    plot_ess_sweep(
        xs=TS,
        xname="T",
        values_by_kernel=ess_vs_T,
        outpath=os.path.join(args.out_dir, "ess_mcmc_vs_T.png"),
        args=args,
    )
    plot_ess_sweep(
        xs=STEPS,
        xname="steps",
        values_by_kernel=ess_vs_steps,
        outpath=os.path.join(args.out_dir, "ess_mcmc_vs_steps.png"),
        args=args,
    )

    plot_param_heatmaps(args)
    # write_summary_csv(args)

    print(f"Saved summary plots and table to: {args.out_dir}")


if __name__ == "__main__":
    main()
