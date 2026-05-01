from typing import Callable

from chex import Array
import jax.numpy as jnp
from jax import vmap



def integrate_dt(a: Array, b: Array, f: Callable, xs: Array):
    """
    Implementation of a left-point Riemann approximation scheme for time integrals
    involving functions of discretised SDE paths.

    Parameters
    ----------
    b:          The integral endpoint
    a:          The integral startpoint
    f:          The function to integrate. Must take (t, x) as args
    xs:         The discretised path, including both endpoints, shape (num, dx)

    Returns
    ----------
    Approximation to the time integral
    """
    num = xs.shape[0] - 1
    dt = (b - a) / num
    ts = a + jnp.arange(num) * dt
    fs = vmap(f)(ts, xs[:-1])
    return jnp.sum(fs * dt)


def integrate_dx(a: Array, b: Array, f: Callable, xs: Array):
    """
    Implementation of a left-point Riemann-Stieltjes approximation scheme for path integrals
    involving functions of discretised SDE paths.

    Parameters
    ----------
    b:          The integral endpoint
    a:          The integral startpoint
    f:          The function to integrate. Must take (t, x) as args
    xs:         The discretised path, including both endpoints, shape (num, dx)

    Returns
    ----------
    Approximation to the path integral
    """
    num = xs.shape[0] - 1
    dt = (b - a) / num
    ts = a + jnp.arange(num) * dt
    dx = xs[1:] - xs[:-1]
    fs = vmap(f)(ts, xs[:-1])
    return jnp.einsum("ni,ni->", fs, dx)


def integrate_dcov_inv(a: Array, b: Array, f: Callable, cov: Callable, xs: Array):
    """
    Implementation of a right-point Riemann-Stieltjes approximation scheme for covariance integrals
    involving functions of discretised SDE paths.

    Parameters
    ----------
    b:          The integral endpoint
    a:          The integral startpoint
    f:          The function to integrate. Must take (t, x) as args
    cov:        The covariance function. Must take (t, x) as args
    xs:         The discretised path, including both endpoints, shape (num, dx)

    Returns
    ----------
    Approximation to the covariance integral
    """
    num = xs.shape[0] - 1
    dt = (b - a) / num
    ts = a + jnp.arange(num + 1) * dt
    
    cov_invs = vmap(lambda t, x: jnp.linalg.inv(cov(t, x)))(ts[:-1], xs[:-1])
    dcov_invs = cov_invs[1:] - cov_invs[:-1]
    
    fs = vmap(f)(ts[1:-1], xs[1:-1])
    return jnp.einsum("ni,nij,nj->", fs, dcov_invs, fs)



def delyonhu(xs: Array, drift: Callable, diffusion: Callable, t0: Array, dt: Array):
    """
    Calculate the logarithm of the Delyon-Hu functional given in Stanton 2025

    Parameters
    ----------
    xs:          (num, dx) The discretised SDE path for t_{k-1} -> t_k
    drift:       The drift function of the SDE. Takes (t, x) as args
    diffusion:   The diffusion function of the SDE. Takes (t, x) as args
    t0:          The starting time
    dt:          The length of time t_k - t_{k-1}
    
    Returns
    -------
    log [ exp{phi_t(v_t)} ]  The log DH function used in the Radon-Nikodym derivative of the potential function
    """

    def _cov(t, x):
        x = jnp.atleast_1d(x)
        sig = diffusion(t, x) * jnp.eye(x.shape[0])
        return sig @ sig.T

    _i1 = lambda t, x: jnp.linalg.inv(_cov(t, x)) @ drift(t, x)
    _i2 = lambda t, x: drift(t, x) @ _i1(t, x)
    _i3 = lambda t, x: (xs[-1] - x) / jnp.sqrt(t0 + dt - t)

    i1 = integrate_dx(t0, t0 + dt, _i1, xs)
    i2 = integrate_dt(t0, t0 + dt, _i2, xs)
    i3 = integrate_dcov_inv(t0, t0 + dt, _i3, _cov, xs)

    return i1 - 0.5*i2 - i3



########################
##        TESTS       ##
########################

def _test_integrate_dt():
    T = 2.0
    num = 10001
    ts = jnp.linspace(0.0, T, num)
    xs = jnp.stack([ts, 2.0 * ts], axis=1)   # (num, 2)

    f = lambda t, x: t + jnp.sum(x)

    val = integrate_dt(0.0, T, f, xs)
    true = 2.0 * T**2

    print("integrate_dt:")
    print("approx =", val)
    print("true   =", true)
    print("error  =", jnp.abs(val - true))



def _test_integrate_dx():
    T = 2.0
    num = 10001
    ts = jnp.linspace(0.0, T, num)
    xs = jnp.stack([ts, 2.0 * ts], axis=1)   # (num, 2)

    f = lambda t, x: x

    val = integrate_dx(0.0, T, f, xs)
    true = 2.5 * T**2

    print("\nintegrate_dx:")
    print("approx =", val)
    print("true   =", true)
    print("error  =", jnp.abs(val - true))


def _test_integrate_dcov_inv():
    T = 2.0
    num = 10001
    ts = jnp.linspace(0.0, T, num)
    xs = jnp.stack([ts, 2.0 * ts], axis=1)   # (num, 2)

    def cov(t, x):
        return (1.0 + t) * jnp.eye(2)

    f = lambda t, x: jnp.ones(2)

    val = integrate_dcov_inv(0.0, T, f, cov, xs)
    true = 2.0 * (1.0 / (1.0 + T) - 1.0)

    print("\nintegrate_dcov_inv:")
    print("approx =", val)
    print("true   =", true)
    print("error  =", jnp.abs(val - true))


if __name__ == "__main__":
    _test_integrate_dt()
    _test_integrate_dx()
    _test_integrate_dcov_inv()