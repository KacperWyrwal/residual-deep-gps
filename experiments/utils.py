from jax import config 
config.update("jax_enable_x64", True)


from jaxtyping import Array, Key, Int
from gpjax.typing import Float
from numpy.random import Generator

import jax 
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial
from jax.scipy.special import gammaln


###
# GENERAL UTILS
### 


def jax_key_to_numpy_generator(key: Key) -> Generator:
    return np.random.default_rng(np.asarray(key._base_array))


def array(x):
    return jnp.array(x, dtype=jnp.float64)


@jax.jit 
def comb(N, k) -> Int:
    return jnp.round(jnp.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1))).astype(jnp.int64)


### 
# SPHERE UTILS
###

@jax.jit
def car_to_sph(car: Float[Array, "N 3"]) -> Float[Array, "N 2"]:
    x, y, z = car[..., 0], car[..., 1], car[..., 2]
    colat = jnp.arccos(z)
    lon = jnp.arctan2(y, x)
    lon = (lon + 2 * jnp.pi) % (2 * jnp.pi)
    return jnp.stack([colat, lon], axis=-1)


@jax.jit
def sph_to_car(sph: Float[Array, "N 2"]) -> Float[Array, "N 3"]:
    colat, lon = sph[..., 0], sph[..., 1]
    z = jnp.cos(colat)
    r = jnp.sin(colat)
    x = r * jnp.cos(lon)
    y = r * jnp.sin(lon)
    return jnp.stack([x, y, z], axis=-1)


@jax.jit
def sphere_expmap(x: Float[Array, "N D"], v: Float[Array, "N D"]) -> Float[Array, "N D"]:
    theta = jnp.linalg.norm(v, axis=-1, keepdims=True)

    t = x + v
    first_order_approx = t / jnp.linalg.norm(t, axis=-1, keepdims=True)
    true_expmap = jnp.cos(theta) * x + jnp.sin(theta) * v / theta

    return jnp.where(
        theta < 1e-12,
        first_order_approx,
        true_expmap,
    )


@jax.jit 
def sphere_to_tangent(x: Float[Array, "N D"], v: Float[Array, "N D"]) -> Float[Array, "N D"]:
    v_x = jnp.sum(x * v, axis=-1, keepdims=True)
    return v - v_x * x


@Partial(jax.jit, static_argnames=("sphere_dim"))
def num_phases_in_frequency(sphere_dim: int, frequency: Int) -> Int:
    l, d = frequency, sphere_dim
    return jnp.where(
        l == 0, 
        jnp.ones_like(l, dtype=jnp.int64), 
        comb(l + d - 2, l - 1) + comb(l + d - 1, l),
    )


# FIXME temporary hack 

from scipy.special import comb as comb_static
from numpy.typing import ArrayLike


def num_phases_in_frequency_static(sphere_dim: int, frequency: ArrayLike) -> ArrayLike:
    l, d = frequency, sphere_dim
    return np.where(
        l == 0, 
        np.ones_like(l, dtype=np.int64), 
        comb_static(l + d - 2, l - 1) + comb_static(l + d - 1, l),
    )


# TODO translate into jax 
def num_phases_to_num_levels(num_phases: int, *, sphere_dim: int) -> int:
    l = 0
    while num_phases > 0:
        num_phases -= num_phases_in_frequency(frequency=l, sphere_dim=sphere_dim)
        l += 1
    return l - 1 if num_phases == 0 else l - 2


def sphere_uniform(sphere_dim: int, n: int, *, key: Key) -> Float[Array, "N D"]:
    x = jax.random.normal(key, (n, sphere_dim + 1))
    return x / jnp.linalg.norm(x, axis=-1, keepdims=True)


def sphere_meshgrid(n_colat=100, n_lon=100, epsilon=1e-32):
    colat = np.linspace(0 + epsilon, np.pi - epsilon, n_colat)
    lon = np.linspace(0, 2 * np.pi, n_lon)
    lon, colat = np.meshgrid(lon, colat)
    return colat, lon


def sphere_uniform_grid(n: int) -> Array:
    """
    Fibonacci lattice on the sphere in Cartesian coordinates. 
    """
    phi = (1 + jnp.sqrt(5)) / 2  # Golden ratio
    
    indices = jnp.arange(n)
    theta = 2 * jnp.pi * indices / phi
    phi = jnp.arccos(1 - 2 * (indices + 0.5) / n)
    return sph_to_car(jnp.column_stack((phi, theta)))


hodge_star_matrix = jnp.array([
    [0.0, 1.0],
    [-1.0, 0.0],
])


@Partial(jax.jit, static_argnames=('min_value', ))
def clip_colatitude(x: Float[Array, "2"], min_value: float) -> Float[Array, "2"]:
    return jax.lax.cond(
        x[0] < min_value,
        lambda: x.at[0].set(min_value),
        lambda: x,
    )


@jax.jit
def tangent_basis(x: Float[Array, "3"]) -> Float[Array, "3"]:
    tb = jax.jacfwd(sph_to_car)(x)
    tb /= jnp.linalg.norm(tb, axis=0, keepdims=True)
    return tb 


@jax.jit
def tangent_basis_normalization_matrix(x: Float[Array, "2"]) -> Float[Array, "2 2"]:
    return jnp.array([
        [1.0, 0.0], 
        [0.0, 1.0 / jnp.sin(x[0])],
    ])


@Partial(jax.jit, static_argnames=("colatitude_min_value", ))
def expmap_sph(x: Float[Array, "2"], v: Float[Array, "2"], colatitude_min_value: float = 1e-12) -> Float[Array, "2"]:
    """
    Exponential map on the sphere taking x in spherical coordinates and v in the 'canonical' coordinate frame. 
    This function internally ensures that the colatitude of x is not too small to avoid nans.
    """
    x = clip_colatitude(x, colatitude_min_value)
    x_prime = sph_to_car(x)
    v_prime = tangent_basis(x) @ v
    return car_to_sph(sphere_expmap(x_prime, v_prime))