"""
PLOT THE PARTICLE FILTERING RESULTS
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

from experiments.ornstein_uhlenbeck.model import get_dynamics
from experiments.ornstein_uhlenbeck.kernels import KernelType

from cd_ssm import bridge
from cd_ssm.utils.printing import ctext


# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=10)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--M", dest="M", type=int, default=5)
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1
parser.add_argument("--mesh-num", type=int, dest="mesh_num", default=50)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--log-var", dest="log_var", type=float, default=0)
parser.add_argument("--phi", dest="phi", type=float, default=0.8)

parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.CSMC)
parser.add_argument("--style", dest="style", type=str, default="guided")

parser.add_argument("--adaptation", dest="adaptation", type=int, default=1)
parser.add_argument("--burnin", dest="burnin", type=int, default=1)
parser.add_argument("--n-samples", dest="n_samples", type=int, default=1)

parser.add_argument("--i", type=int, default=0)

parser.add_argument("--grouped", dest="grouped", action="store_true")
parser.set_defaults(grouped=False)

args = parser.parse_args()


###############################
#  SINGLE ANALYSIS FUNCTIONS  #
###############################

# --- args ---
RHO = 0.5
kernel_type = KernelType(args.kernel)

SIGMA = 10 ** (args.log_var / 2)
PHI = args.phi
DRIFT, DIFFUSION = get_dynamics(PHI, SIGMA)
DT = args.T / args.steps
Ts = np.arange(args.steps) * DT

# --- functions ---
def plot_mean_paths(data, dirpath, dim=0):
    """
    Plot one panel per MCMC chain showing:
    - true latent path (black dashed)
    - mean smoothing sample path (blue)
    - observations (red scatter)

    Parameters
    ----------
    data : dict-like
        Must contain:
        - true_xs:    shape (..., T, D)
        - ys:         shape (..., T, D)
        - means:      shape (..., K, T, M, D)
        - std_devs:   shape (..., K, T, M, D)
    dirpath : str
        Directory where the figure will be saved.
    i : int, optional
        Index of the dataset/trajectory to plot.
    dim : int, optional
        State dimension to plot.

    Returns
    -------
    None
    """

    # path reconstructor
    _to_path = jax.vmap(lambda u, ep, e, t: bridge.to_path(DIFFUSION, u, ep, e, t, DT))
    to_path = jax.vmap(lambda u, ep, e, t: _to_path(u, ep, e, t), in_axes=(0, 0, 0, None))

    # truth
    true_us = np.asarray(data["true_us"][args.i])[None, ...]   # (1, T, M, D)
    true_es = np.asarray(data["true_es"][args.i])[None, ...]   # (1, T, D)
    ys = np.asarray(data["ys"][args.i])             # (T, D)
    true_paths = to_path(true_us[:, 1:], true_es[:, :-1], true_es[:, 1:], Ts[1:])

    # sampled paths
    mean_paths = data["means"][args.i][..., dim]  # (K, T, M)
    std_devs = data["std_devs"][args.i][..., dim] # (K, T, M)
    K, T, M = mean_paths.shape

    # flatten each chain's path
    t_fine = np.concatenate([t + np.arange(1, M + 1) / M for t in range(T)])
    mean_path_vals = mean_paths.reshape(K, -1)
    std_devs = std_devs.reshape(K, -1)
    true_path_vals = true_paths[..., dim].reshape(1, -1)

    # layout
    M = min(K, 10)
    fig, axes = plt.subplots(M, 1, figsize=(15, 3.5 * M), squeeze=True)
    t_obs = np.arange(T)

    for k in range(M):
        ax = axes[k]
        mean_k = mean_path_vals[k]
        std_k = std_devs[k]
        ax.plot(t_fine, mean_k, label="Path", color="blue")
        ax.fill_between(t_fine, mean_k - std_k, mean_k + std_k, alpha=0.15, label=r"Mean $\pm$ 1 std. dev.", color="blue")

        ax.plot(t_fine, true_path_vals[0], linestyle="--", label="True path", color="black")
        ax.scatter(t_obs, ys[1:, dim], marker="x", c="red", label="Observations",)

        ax.set_title(f"Chain {k}")
        ax.set_xlabel("t")
        ax.set_ylabel(f"Dimension {dim}")
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for k in range(K, len(axes)):
        axes[k].axis("off")

    # single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    outpath = os.path.join(dirpath, "paths.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_traces(data, dirpath):
    """
    Plot one panel per MCMC chain showing:
    - trace plots of a one parameter ie x[t, dim] over n_samples for 3 different ts (0, T/2, T)

    Parameters
    ----------
    data : dict-like
        Must contain:
        - traces:    shape (..., n_samples, K, 3, 1)
    dirpath : str
        Directory where the figure will be saved.
    i : int, optional
        Index of the dataset/trajectory to plot.
    dim : int, optional
        State dimension to plot.

    Returns
    -------
    None
    """
    # path reconstructor
    _to_path = jax.vmap(lambda u, ep, e, t: bridge.to_path(DIFFUSION, u, ep, e, t, DT))
    to_path = jax.vmap(lambda u, ep, e, t: _to_path(u, ep, e, t), in_axes=(0, 0, 0, None))

    # truth
    true_us = np.asarray(data["true_us"][args.i])[None, ...]   # (1, T, M, D)
    true_es = np.asarray(data["true_es"][args.i])[None, ...]   # (1, T, D)
    true_paths = to_path(true_us[:, 1:], true_es[:, :-1], true_es[:, 1:], Ts[1:])

    traces = data["traces"][args.i]   # (n_samples, K, 3, 1)
    N, K = traces.shape[:2]
    times = ["0", "steps/2", "steps"]
    true_vals = [true_paths[0, 0, 0, 0], true_paths[0, args.steps // 2, 0, 0], true_paths[0, args.steps - 2, 0, 0]]

    M = (min(K, 10))
    fig, ax = plt.subplots(M, 3, figsize=(25, 5*M))
    for k in range(M):
        for i in range(3):
            _trace = traces[:, k, i]
            _truth = true_vals[i]
            ax[k, i].plot(_trace[::10], label="Thinned trace")
            ax[k, i].axhline(y=_truth, color="red", linestyle="--", linewidth=1.5, label="True value",)
            ax[k, i].set_title(f"Chain {k}, time: {times[i]}, dim: 0")
            ax[k, i].legend()
    plt.tight_layout()

    outpath = os.path.join(dirpath, "traces.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ess(data, dirpath):
    """
    Plot one panel per MCMC chain showing:
    - thinned set of ESS timeseries for every 10 samples
    - average ESS trace over sample time

    Parameters
    ----------
    data : dict-like
        Must contain:
        - ess:    shape (..., n_samples, K, steps)
    dirpath : str
        Directory where the figure will be saved.
    i : int, optional
        Index of the dataset/trajectory to plot.
    dim : int, optional
        State dimension to plot.

    Returns
    -------
    None
    """
    ess = data["ess"][args.i]  # (N, K, steps)
    N, K, steps = ess.shape

    M = min(K, 10)
    fig, ax = plt.subplots(M, 2, figsize=(15, 5*M))
    for k in range(M):
        _ess_thinned = ess[::10, k, :]
        ax[k, 0].plot(_ess_thinned, alpha=0.1, color="blue")
        ax[k, 0].set_title("Various ESS series from samples")
        ax[k, 0].axhline(y=args.N+1, color="red", linestyle="--", linewidth=1.5, label="N particles",)
        ax[k, 0].legend()

        avg_ess = ess[:, k].mean(axis=0)  # (steps,)
        ax[k, 1].plot(avg_ess, color="black", linestyle="--")
        ax[k, 1].set_title("Average ESS over samples")
        ax[k, 1].axhline(y=args.N+1, color="red", linestyle="--", linewidth=1.5, label="N particles",)
        ax[k, 1].legend()
    
    plt.tight_layout()

    outpath = os.path.join(dirpath, "ess.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

########################
#  load data function  #
########################

def load_data(kernel, style, D, mesh_num, steps):
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
    dirpath = f"results/{experiment_name}"
    if not os.path.exists(dirpath):
        print(ctext("No such experiment exists", "yellow"))
        print(experiment_name)
        return None, None 
        # exit()

    data = np.load(f"{dirpath}/data.npz")
    return data, dirpath
    
data, dirpath = load_data(
    kernel_type,
    args.style,
    args.D,
    args.mesh_num,
    args.steps,
)
plot_mean_paths(data, dirpath)
plot_traces(data, dirpath)
plot_ess(data, dirpath)
