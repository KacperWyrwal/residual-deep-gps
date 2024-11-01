# TODO Credit that repo 
from jax import config 
config.update("jax_enable_x64", True)


from typing import Callable
from jaxtyping import Array
from gpjax.typing import Float 


import jax 
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial
from gpjax.base import static_field, Module
from dataclasses import dataclass
from pathlib import Path
from abc import abstractmethod
from utils import sph_to_car, clip_colatitude, hodge_star_matrix, tangent_basis_normalization_matrix


class FundamentalSystemNotPrecomputedError(ValueError):

    def __init__(self, dimension: int):
        message = f"Fundamental system for dimension {dimension} has not been precomputed."
        super().__init__(message)


def fundamental_set_loader(dimension: int) -> Callable[[int], Array]:
    load_dir = Path("../fundamental_system")
    file_name = load_dir / f"fs_{dimension}D.npz"

    cache = {}
    if file_name.exists():
        with np.load(file_name) as f:
            cache = {k: v for (k, v) in f.items()}
    else:
        raise FundamentalSystemNotPrecomputedError(dimension)

    def load(degree: int) -> Array:
        key = f"degree_{degree}"
        if key not in cache:
            raise ValueError(f"key: {key} not in cache.")
        return cache[key]

    return load


@Partial(jax.jit, static_argnames=('max_ell', 'alpha',))
def gegenbauer(x: Float[Array, "N D"], max_ell: int, alpha: float = 0.5) -> Float[Array, "N L"]:
    """
    Evaluates gegenbauer polynomials up to max_ell order.
    """
    C_0 = jnp.ones_like(x, dtype=x.dtype)
    C_1 = 2 * alpha * x
    
    res = jnp.empty((*x.shape, max_ell + 1), dtype=x.dtype)
    res = res.at[..., 0].set(C_0)

    def step(n: int, res_and_Cs: tuple[Float, Float, Float]) -> tuple[Float, Float, Float]:
        res, C, C_prev = res_and_Cs
        C, C_prev = (2 * x * (n + alpha - 1) * C - (n + 2 * alpha - 2) * C_prev) / n, C
        res = res.at[..., n].set(C)
        return res, C, C_prev
    
    return jax.lax.cond(
        max_ell == 0,
        lambda: res,
        lambda: jax.lax.fori_loop(2, max_ell + 1, step, (res.at[..., 1].set(C_1), C_1, C_0))[0],
    )


@dataclass
class SphericalHarmonics(Module):
    """
    Spherical harmonics inducing features for sparse inference in Gaussian processes.

    The spherical harmonics, Yₙᵐ(·) of frequency n and phase m are eigenfunctions on the sphere and,
    as such, they form an orthogonal basis.

    To construct the harmonics, we use a a fundamental set of points on the sphere {vᵢ}ᵢ and compute
    b = {Cᵅₙ(<vᵢ, x>)}ᵢ. b now forms a complete basis on the sphere and we can orthogoalise it via
    a Cholesky decomposition. However, we only need to run the Cholesky decomposition once during
    initialisation.

    Attributes:
        num_frequencies: The number of frequencies, up to which, we compute the harmonics.

    Returns:
        An instance of the spherical harmonics features.
    """

    max_ell: int = static_field()
    sphere_dim: int = static_field()
    alpha: float = static_field(init=False)
    orth_basis: Array = static_field(init=False)
    Vs: list[Array] = static_field(init=False)
    num_phases_per_frequency: Float[Array, " L"] = static_field(init=False)
    num_phases: int = static_field(init=False)


    @property
    def levels(self):
        return jnp.arange(self.max_ell + 1, dtype=jnp.int32)
    

    def __post_init__(self) -> None:
        """
        Initialise the parameters of the spherical harmonic features and return a `Param` object.

        Returns:
            None
        """
        dim = self.sphere_dim + 1

        # Try loading a pre-computed fundamental set.
        fund_set = fundamental_set_loader(dim)

        # initialise the Gegenbauer lookup table and compute the relevant constants on the sphere.
        self.alpha = (dim - 2.0) / 2.0

        # initialise the parameters Vs. Set them to non-trainable if we do not truncate the phase.
        self.Vs = [fund_set(n) for n in self.levels]

        # pre-compute and save the orthogonal basis 
        self.orth_basis = self._orthogonalise_basis()


        # set these things instead of computing every time 
        self.num_phases_per_frequency = [v.shape[0] for v in self.Vs]
        self.num_phases = sum(self.num_phases_per_frequency)


    @property
    def Ls(self) -> list[Array]:
        """
        Alias for the orthogonal basis at every frequency.
        """
        return self.orth_basis

    def _orthogonalise_basis(self) -> None:
        """
        Compute the basis from the fundamental set and orthogonalise it via Cholesky decomposition.
        """
        alpha = self.alpha
        levels = jnp.split(self.levels, self.max_ell + 1)
        const = alpha / (alpha + self.levels.astype(jnp.float64))
        const = jnp.split(const, self.max_ell + 1)

        def _func(v, n, c):
            x = jnp.matmul(v, v.T)
            B = c * self.custom_gegenbauer_single(x, ell=n[0], alpha=self.alpha)
            L = jnp.linalg.cholesky(B + 1e-16 * jnp.eye(B.shape[0], dtype=B.dtype))
            return L

        return jax.tree.map(_func, self.Vs, levels, const)

    def custom_gegenbauer_single(self, x, ell, alpha):
        return gegenbauer(x, self.max_ell, alpha)[..., ell]

    @jax.jit
    def polynomial_expansion(self, X: Float[Array, "N D"]) -> Float[Array, "M N"]:
        """
        Evaluate the polynomial expansion of an input on the sphere given the harmonic basis.

        Args:
            X: Input Array.

        Returns:
            The harmonics evaluated at the input as a polynomial expansion of the basis.
        """
        levels = jnp.split(self.levels, self.max_ell + 1)

        def _func(v, n, L):
            vxT = jnp.dot(v, X.T)
            zonal = self.custom_gegenbauer_single(vxT, ell=n[0], alpha=self.alpha)
            harmonic = jax.lax.linalg.triangular_solve(L, zonal, left_side=True, lower=True)
            return harmonic

        harmonics = jax.tree.map(_func, self.Vs, levels, self.Ls)
        return jnp.concatenate(harmonics, axis=0)
    
    def __eq__(self, other: "SphericalHarmonics") -> bool:
        """
        Check if two spherical harmonic features are equal.

        Args:
            other: The other spherical harmonic features.

        Returns:
            A boolean indicating if the two features are equal.
        """
        # Given the first two parameters, the rest are deterministic. 
        # The user must not mutate all other fields, but that is not enforced for now.
        return (
            self.max_ell == other.max_ell 
            and self.sphere_dim == other.sphere_dim 
        )    
    

@dataclass 
class AbstractSphericalHarmonicFields(Module):
    max_ell: int = static_field(10)
    sphere_dim: int = static_field(2)
    colatitude_min_value: float = static_field(1e-12)
    spherical_harmonics: SphericalHarmonics = static_field(init=False)
    num_phases_per_frequency: Float[Array, "L"] = static_field(init=False)
    num_phases: int = static_field(init=False)
    num_fields: int = static_field(init=False)

    def __post_init__(self) -> None:
        self.spherical_harmonics = SphericalHarmonics(max_ell=self.max_ell, sphere_dim=self.sphere_dim)
        num_phases_per_frequency = self.spherical_harmonics.num_phases_per_frequency[1:]
        self.num_phases_per_frequency = jnp.array(num_phases_per_frequency)
        self.num_phases = sum(num_phases_per_frequency)

    @jax.jit
    def _sph_polynomial_expansion(self, x: Float[Array, "2"]) -> Float[Array, "2"]:
        ells = jnp.arange(1, self.max_ell + 1)
        lambda_ells = ells * (ells + 1) 
        normalization_term = jnp.repeat(
            jnp.sqrt(lambda_ells),
            self.num_phases_per_frequency,
            total_repeat_length=self.num_phases
        )
        return self.spherical_harmonics.polynomial_expansion(sph_to_car(x))[1:] / normalization_term

    @jax.jit
    def eigenfields(self, x: Float[Array, "2"]) -> Float[Array, "2"]:
        x = clip_colatitude(x, self.colatitude_min_value)
        Nx = tangent_basis_normalization_matrix(x)
        return jax.jacfwd(self._sph_polynomial_expansion)(x) @ Nx

    @abstractmethod
    def __call__(self, x: Float[Array, "N 2"]) -> Float[Array, "N 2"]:
        pass 
    
    def __eq__(self, other: "AbstractSphericalHarmonicFields") -> bool:
        return self.max_ell == other.max_ell and self.sphere_dim == other.sphere_dim and self.colatitude_min_value == other.colatitude_min_value
    

@dataclass 
class SphericalHarmonicFields(AbstractSphericalHarmonicFields):

    def __post_init__(self) -> None:
        super().__post_init__()
        self.num_fields = 2 * self.num_phases

    @jax.jit
    def __call__(self, x: Float[Array, "2"]) -> Float[Array, "2I 2"]:
        """
        Returns curl-free and divergence-free fields concatenated.
        """
        H = hodge_star_matrix
        v = self.eigenfields(x) # [I 2]
        return jnp.concat([v, v @ H], axis=-2)
