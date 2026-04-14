"""
TODO:
    1. Write the pcn proposal kernel  - CHECK THIS
    2. Implement the log_potential for pcn proposal
    3. Write out the pcn langevin proposal kernel
"""

from chex import Array, PRNGKey
import jax
import jax.numpy as jnp

from cd_ssm import brownian


def propose(
        key: PRNGKey, 
        xp: Array, 
        rho: Array, 
        dts: Array, 
        N: Array
    ):
    """
    Propose N new paths according to:
         x = rho*xp + sqrt(1 - rho^2) * W(R^D)
    where W is a sample from the Weiner law on R^D

    Parameters
    ----------
    key: RNG
    xp:  Array  (T, mesh, D)
    rho: Array
    dts: Array  (mesh - 1) The time delta between each step
    N:   Array

    Returns
    -------
    x:   Array (T, N, mesh, D)
    """

    T, mesh, D = xp.shape
    xp = jnp.repeat(xp[:, None, ...], N, axis=1)
    x = brownian.simulate(key, xp, dts, mesh, N)
    return rho * x + jnp.sqrt(1 - jnp.square(rho)) * x


def log_potential():
    pass