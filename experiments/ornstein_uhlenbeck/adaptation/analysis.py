"""
PLOT THE ADAPTATION RESULTS
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
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1
parser.add_argument("--mesh-num", type=int, dest="mesh_num", default=50)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--adaptation", dest="adaptation", type=int, default=1000)
parser.add_argument("--target", dest="target", type=int, default=75)

parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.PCN)
parser.add_argument("--style", dest="style", type=str, default="na")

parser.add_argument("--i", type=int, default=0)

parser.add_argument("--grouped", dest="grouped", action="store_true")
parser.set_defaults(grouped=False)

args = parser.parse_args()

kernel_type = KernelType(args.kernel)

def plot_adaptation(data, dirpath):
    
    rho_hist = data["rho_hist"][args.i]
    delta_hist = data["delta_hist"][args.i]
    rho_ar_hist = data["rho_acc_rates_hist"][args.i]
    delta_ar_hist = data["delta_acc_rates_hist"][args.i]

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    # print(rho_hist)
    # print(delta_hist)
    
    ax[0, 0].plot(rho_hist)
    ax[0, 0].set_title("Sequence of rhos during adaptation")
    
    ax[0, 1].plot(rho_ar_hist)
    ax[0, 1].set_title("Sequence of acceptance rates after rho adaptation")

    ax[1, 0].plot(delta_hist)
    ax[1, 0].set_title("Sequence of deltas during adaptation")
    
    ax[1, 1].plot(delta_ar_hist)
    ax[1, 1].set_title("Sequence of acceptance rates after delta adaptation")

    plt.tight_layout()
    plt.savefig(f"{dirpath}/adaptation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


##############################
#  Group analysis functions  #
##############################
DS = (1, 5, 10, 20, 30, 40, 50, 75, 100, )


def plot_adaptation_with_d(dirpath, t: int = 10, steps: int = 100, mesh: int = 50):

    rhos = []
    deltas = []
    
    for j, d in enumerate(DS):
        data, _ = load_data(d, mesh, steps)
        
        if data is None:
            rhos.append(np.nan)
            deltas.append(np.nan)

        else:
            rho_hist = data["rho_hist"]  # (K, iters, 1))
            delta_hist = data["delta_hist"]
            fin_rho = rho_hist[:, -1, 0][0]
            fin_delta = delta_hist[:, -1, 0][0]
            rhos.append(fin_rho)
            deltas.append(fin_delta)

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    
    print(rhos[-1])
    print(deltas[-1])
    ref1 = np.log(np.array(DS))
    ref2 = np.array(DS) **(1/3)
    rhos = np.asarray(rhos)

    ax[0].plot(ref1, np.log(1 - rhos), label="rho")
    # ax[0].plot(DS, ref1, label="D")
    # ax[0].plot(DS, ref2, label="D**1/3")
    ax[0].set_ylabel("Final rho after adaptation")
    ax[0].set_xlabel("D")
    ax[0].legend()

    ax[1].plot(ref1, np.log(deltas), label="delta")
    # ax[1].plot(DS, - ref1, label="1/D")
    # ax[1].plot(DS, 1/ ref2, label="1/(D**1/3)")
    ax[1].set_ylabel("Final delta after adaptation")
    ax[1].set_xlabel("D")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(f"{dirpath}/adaptation_vs_d.png", dpi=200, bbox_inches="tight")
    plt.close()


def load_data(D: int, mesh_num: int, steps: int):
    """ Load data for a given number of particles N"""
    experiment_name = "kernel={},style={},adaptation={},target={},D={},N={},mesh-num={},steps={},seed={}"
    experiment_name = experiment_name.format(
        kernel_type.name,
        args.style,
        args.adaptation,
        args.target,
        D,
        args.N,
        mesh_num,
        steps,
        args.seed
    )

    dirpath = f"results/{experiment_name}"
    if not os.path.exists(dirpath):
        print(ctext("No such experiment exists", "yellow"))
        print(experiment_name)
        return None, None

    data = np.load(f"{dirpath}/data.npz")
    return data, dirpath

if not args.grouped:
    data, dirpath = load_data(args.D, args.mesh_num, args.steps)
    plot_adaptation(data, dirpath)

else:
    dirpath = "results"
    plot_adaptation_with_d(dirpath)