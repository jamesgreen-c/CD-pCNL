"""
PLOT THE PARTICLE FILTERING RESULTS
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

from experiments.lgssm.model import get_dynamics
from experiments.lgssm.kernels import KernelType

from cd_ssm import bridge
from cd_ssm.utils.printing import ctext


# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=10)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1
parser.add_argument("--mesh-num", type=int, dest="mesh_num", default=100)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--log-var", dest="log_var", type=float, default=0)
parser.add_argument("--phi", dest="phi", type=float, default=0.8)

parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.CSMC)
parser.add_argument("--style", dest="style", type=str, default="guided")

parser.add_argument("--i", type=int, default=0)

args = parser.parse_args()

RHO = 0.5
kernel_type = KernelType(args.kernel)

SIGMA = 10 ** (args.log_var / 2)
PHI = args.phi
DRIFT, DIFFUSION = get_dynamics(PHI, SIGMA)
DT = args.T / args.steps
Ts = np.arange(args.steps) * DT


def plot_particles(data, dirpath, dim=0):
    """
    Plot one panel per MCMC chain showing:
    - Plot the smoothing particles (noise, endpoint)

    Parameters
    ----------
    data : dict-like
        Must contain:
        - us:      shape (..., K, T, M, D)
        - es:      shape (..., K, T, D)
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
    
    us = np.asarray(data["us"][args.i])             # (K, T, M, D)
    es = np.asarray(data["es"][args.i])             # (K, T, D)

    K, T, M, D = us.shape

    t_obs = np.arange(T)
    t_fine = np.concatenate([t + np.arange(1, M + 1) / M for t in range(T)])

    fig, axes = plt.subplots(K, 1, figsize=(14, 3.5 * K), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for k, ax in enumerate(axes):
        sample_path = us[k, :, :, dim].reshape(T * M)
        ax.plot(t_fine, sample_path, color="blue", alpha=0.9, linewidth=1.5, label="Smoothing sample")
        ax.scatter(t_obs, es[k, :, dim], color="red", marker="x", s=30, label="End points")
        ax.set_ylabel(f"Chain {k}")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{dirpath}/particles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_paths(data, dirpath, dim=0):
    """
    Plot one panel per MCMC chain showing:
    - true latent path (black dashed)
    - smoothing sample path (blue)
    - observations (red scatter)

    Parameters
    ----------
    data : dict-like
        Must contain:
        - true_xs: shape (..., T, D)
        - ys:      shape (..., T, D)
        - xs:      shape (..., K, T, M, D)
        - us:      shape (..., K, T, M, D)
        - es:      shape (..., K, T, D)
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

    # truth
    true_us = np.asarray(data["true_us"][args.i])[None, ...]   # (1, T, M, D)
    true_es = np.asarray(data["true_es"][args.i])[None, ...]   # (1, T, D)
    ys = np.asarray(data["ys"][args.i])             # (T, D)

    # particles
    us = np.asarray(data["us"][args.i])             # (K, T, M, D)
    es = np.asarray(data["es"][args.i])             # (K, T, D)

    _to_path = jax.vmap(lambda u, ep, e, t: bridge.to_path(DIFFUSION, u, ep, e, t, DT))
    to_path = jax.vmap(lambda u, ep, e, t: _to_path(u, ep, e, t), in_axes=(0, 0, 0, None))
    paths = to_path(us[:, 1:], es[:, :-1], es[:, 1:], Ts[1:])
    true_paths = to_path(true_us[:, 1:], true_es[:, :-1], true_es[:, 1:], Ts[1:])

    K, Tm1, M, D = paths.shape
    T = Tm1 + 1

    # Flatten each chain's path and prepend the t=0 point from es[:, 0]
    t_fine = np.concatenate([[0.0], np.concatenate([t + np.arange(1, M + 1) / M for t in range(T - 1)])])
    path_vals = np.concatenate([es[:, 0, dim][:, None], paths[..., dim].reshape(K, -1)], axis=1)  # (K, 1 + (T-1) * M)
    true_path_vals = np.concatenate([true_es[:, 0, dim][:, None], true_paths[..., dim].reshape(1, -1)], axis=1)  # (K, 1 + (T-1) * M)

    # Layout
    fig, axes = plt.subplots(K, 1, figsize=(15, 3.5 * K), squeeze=True)
    t_obs = np.arange(T)

    for k in range(K):
        ax = axes[k]
        ax.plot(t_fine, path_vals[k], label="Path",)
        ax.plot(t_fine, true_path_vals[0], linestyle="--", label="True u's",)
        ax.scatter(t_obs, true_es[0, :, dim], marker="o", label="True e's",)
        ax.scatter(t_obs, ys[:, dim], marker="x", c="red", label="Observations",)

        ax.set_title(f"Chain {k}")
        ax.set_xlabel("t")
        ax.set_ylabel(f"Dimension {dim}")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for k in range(K, len(axes)):
        axes[k].axis("off")

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    outpath = os.path.join(dirpath, "paths.png")
    fig.savefig(f"{dirpath}/paths.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_esjd(data, dirpath):
    """
    
    """
    esjd_us = data["esjd_us"][args.i]   # (T, )  
    esjd_es = data["esjd_es"][args.i]   # (T, )
    
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[0].plot(Ts, esjd_us)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("ESJD")
    ax[0].set_title("ESJD for driving Brownian motions (u's)")

    ax[1].plot(Ts, esjd_es)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("ESJD")
    ax[1].set_title("ESJD for end points (e's)")

    plt.tight_layout()
    fig.savefig(f"{dirpath}/esjd.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_data():
    """ Load data for a given number of particles N"""
    experiment_name = "kernel={},style={},rho={},D={},N={},mesh-num={},steps={},seed={}"
    experiment_name = experiment_name.format(
        kernel_type.name,
        args.style,
        RHO,
        args.D,
        args.N,
        args.mesh_num,
        args.steps,
        args.seed
    )

    dirpath = f"results/{experiment_name}"
    if not os.path.exists(dirpath):
        print(ctext("No such experiment exists", "yellow"))
        print(experiment_name)
        exit()

    data = np.load(f"{dirpath}/data.npz")
    return data, dirpath

data, dirpath = load_data()
plot_particles(data, dirpath)
plot_paths(data, dirpath)
plot_esjd(data, dirpath)