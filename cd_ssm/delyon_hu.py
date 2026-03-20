from typing import Callable

from chex import Array
import jax.numpy as jnp

from cd_ssm.utils.numerics import integrate_dt, integrate_dx, integrate_dcov_inv



def delyonhu(xs: Array, drift: Callable, diffusion: Callable, dt: Array):
    """
    Calculate the logarithm of the Delyon-Hu functional given in Stanton 2025

    Parameters
    ----------
    xs:          (num, dx) The discretised SDE path for t_{k-1} -> t_k
    drift:       The drift function of the SDE. Takes (t, x) as args
    diffusion:   The diffusion function of the SDE. Takes (t, x) as args
    dt:          The length of time t_k - t_{k-1}
    
    Returns
    -------
    log [ exp{phi_t(v_t)} ]  The log DH function used in the Radon-Nikodym derivative of the potential function
    """

    def _cov(t, x):
        sig = diffusion(t, x)
        return sig @ sig.T

    _i1 = lambda t, x: jnp.linalg.inv(_cov(t, x)) @ drift(t, x)
    _i2 = lambda t, x: drift(t, x) @ _i1(t, x)
    _i3 = lambda t, x: (xs[-1] - x) / jnp.sqrt(dt - t)

    i1 = integrate_dx(dt, 0, _i1, xs)
    i2 = integrate_dt(dt, 0, _i2, xs)
    i3 = integrate_dcov_inv(dt, 0, _i3, _cov, xs)

    return i1 - 0.5*i2 - i3