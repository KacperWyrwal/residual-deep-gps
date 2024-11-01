from jax import config 
config.update("jax_enable_x64", True)


from typing import Callable
from jaxtyping import Array, Key, Int, Num 
from gpjax.typing import Float, ScalarFloat 


import jax 
import jax.numpy as jnp
import numpy as np
import pandas as pd
import gpjax 
from jax.tree_util import Partial
from jax.scipy.special import gammaln
from gpjax.base import static_field, param_field, Module
from dataclasses import dataclass, InitVar
import tensorflow_probability.substrates.jax.bijectors as tfb
from spherical_harmonics import gegenbauer, SphericalHarmonics
from utils import num_phases_in_frequency, tangent_basis_normalization_matrix, clip_colatitude, hodge_star_matrix




### EUCLIDEAN MATERN 32  
@jax.jit 
def grad_safe_norm(z: Float[Array, "... D"]) -> Float[Array, "..."]:
    return jnp.sqrt(jnp.clip(jnp.sum(jnp.square(z), axis=-1), min=1e-36))


@jax.jit 
def euclidean_matern32_kernel(z: Float, variance: Float) -> Float:
    return variance * (1 + jnp.sqrt(3) * z) * jnp.exp(-jnp.sqrt(3) * z)


@dataclass 
class EuclideanMaternKernel32(Module):
    num_inputs: int = static_field(init=True)
    kappa: Float = param_field(jnp.array([1.0]), bijector=tfb.Softplus())
    variance: Float = param_field(jnp.array(1.0), bijector=tfb.Softplus())

    def __post_init__(self):
        self.kappa = jnp.asarray(self.kappa, dtype=jnp.float64)
        self.variance = jnp.asarray(self.variance, dtype=jnp.float64)

        assert self.kappa.shape[0] == self.num_inputs or self.kappa.shape[0] == 1

    @jax.jit
    def prepare_inputs(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, ""]:
        return grad_safe_norm((x - y) / self.kappa)

    @jax.jit 
    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, ""]:
        return euclidean_matern32_kernel(self.prepare_inputs(x, y), self.variance)


@dataclass 
class MultioutputEuclideanMaternKernel32(Module):
    num_inputs: int = static_field()
    num_outputs: int = static_field()
    kappa: ScalarFloat = param_field(jnp.array([1.0]), bijector=tfb.Softplus())
    variance: ScalarFloat = param_field(jnp.array([1.0]), bijector=tfb.Softplus())

    def _validate_params(self) -> None:
        # float64 for numerical stability
        self.kappa = jnp.asarray(self.kappa, dtype=jnp.float64)
        self.variance = jnp.asarray(self.variance, dtype=jnp.float64)        

        # shape for multioutput
        kappa_shape = jnp.broadcast_shapes(self.kappa.shape, (self.num_outputs, 1))
        self.kappa = jnp.broadcast_to(self.kappa, kappa_shape)
        self.variance = jnp.broadcast_to(self.variance, (self.num_outputs,))

        assert self.kappa.shape[1] == self.num_inputs or self.kappa.shape[1] == 1

    def __post_init__(self):
        self._validate_params()

    @jax.jit
    def prepare_inputs(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, "O"]:
        return grad_safe_norm((x - y) / self.kappa)

    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, ""]:
        return jax.vmap(euclidean_matern32_kernel)(self.prepare_inputs(x, y), self.variance)



### MANIFOLD MATERN 


@Partial(jax.jit, static_argnames=("max_ell", "sphere_dim"))
def sphere_addition_theorem(x: Float[Array, "D"], y: Float[Array, "D"], *, max_ell: int, sphere_dim: int) -> Float:
    alpha = (sphere_dim - 1) / 2.0
    c1 = num_phases_in_frequency(sphere_dim=sphere_dim, frequency=jnp.arange(max_ell + 1))
    c2 = gegenbauer(1.0, max_ell=max_ell, alpha=alpha)
    Pz = gegenbauer(jnp.dot(x, y), max_ell=max_ell, alpha=alpha)
    return c1 / c2 * Pz


def addition_theorem_scalar_kernel(spectrum: Float[Array, "I"], z: Float[Array, "I"]) -> Float[Array, ""]:
    return jnp.dot(spectrum, z)


# TODO rename to sphere_matern_spectrum
@Partial(jax.jit, static_argnames=('dim',))
def matern_spectrum(ell: Float, kappa: Float, nu: Float, variance: Float, dim: int) -> Float:
    lambda_ells = ell * (ell + dim - 1)
    log_Phi_nu_ells = -(nu + dim / 2) * jnp.log1p((lambda_ells * kappa**2) / (2 * nu))
    
    # Subtract max value for numerical stability
    max_log_Phi = jnp.max(log_Phi_nu_ells)
    Phi_nu_ells = jnp.exp(log_Phi_nu_ells - max_log_Phi)
    
    # Normalize the density, so that it sums to 1
    num_harmonics_per_ell = num_phases_in_frequency(frequency=ell, sphere_dim=dim)
    normalizer = jnp.dot(num_harmonics_per_ell, Phi_nu_ells)
    return variance * Phi_nu_ells / normalizer


@dataclass
class SphereMaternKernel(Module):
    sphere_dim: int = static_field(2)
    kappa: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    nu: ScalarFloat = param_field(jnp.array(1.5), bijector=tfb.Softplus())
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    max_ell: int = static_field(25)
    sample_max_ell: int = static_field(10)
    spherical_harmonics: SphericalHarmonics = static_field(init=False)

    def __post_init__(self):
        self.kappa = jnp.asarray(self.kappa, dtype=jnp.float64)
        self.nu = jnp.asarray(self.nu, dtype=jnp.float64)
        self.variance = jnp.asarray(self.variance, dtype=jnp.float64)
        self.spherical_harmonics = SphericalHarmonics(max_ell=self.sample_max_ell, sphere_dim=self.sphere_dim)

    @property 
    def ells(self):
        return jnp.arange(self.max_ell + 1, dtype=jnp.float64)
    
    def spectrum(self) -> Num[Array, "I"]:
        return matern_spectrum(self.ells, self.kappa, self.nu, self.variance, dim=self.sphere_dim)

    @jax.jit 
    def from_spectrum(self, spectrum: Float[Array, "M"], x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, ""]:
        return addition_theorem_scalar_kernel(
            spectrum, 
            sphere_addition_theorem(x, y, max_ell=self.max_ell, sphere_dim=self.sphere_dim)
        )
    
    @jax.jit 
    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, ""]:
        return self.from_spectrum(self.spectrum(), x, y)
    
    @jax.jit 
    def phi(self, x: Float[Array, "D"]) -> Float[Array, "F"]:
        spectrum = self.spectrum()
        weights = jnp.repeat(
            jnp.sqrt(spectrum), 
            num_phases_in_frequency(self.ells, sphere_dim=self.sphere_dim),
        )
        phi = self.spherical_harmonics.polynomial_expansion(x)
        return phi.T @ weights
    

@dataclass 
class MultioutputSphereMaternKernel(Module):
    num_outputs: int = static_field()
    sphere_dim: int = static_field(2)
    kappa: ScalarFloat = param_field(jnp.array([1.0]), bijector=tfb.Softplus())
    nu: ScalarFloat = param_field(jnp.array([1.5]), bijector=tfb.Softplus())
    variance: ScalarFloat = param_field(jnp.array([1.0]), bijector=tfb.Softplus())
    max_ell: int = static_field(25)
    total_repeat_length: int = static_field(init=False)
    spherical_harmonics: SphericalHarmonics = static_field(init=False)
    # spherical_harmonics: SphericalHarmonics = param_field(init=False, trainable=False)

    def _validate_params(self) -> None:
        # float64 for numerical stability
        self.kappa = jnp.asarray(self.kappa, dtype=jnp.float64)
        self.nu = jnp.asarray(self.nu, dtype=jnp.float64)
        self.variance = jnp.asarray(self.variance, dtype=jnp.float64)

        # shape for multioutput
        self.kappa = jnp.broadcast_to(self.kappa, (self.num_outputs,))
        self.nu = jnp.broadcast_to(self.nu, (self.num_outputs,))
        self.variance = jnp.broadcast_to(self.variance, (self.num_outputs,))

    def __post_init__(self):
        self._validate_params()
        from utils import num_phases_in_frequency_static

        self.total_repeat_length = int(sum(num_phases_in_frequency_static(frequency=np.arange(self.max_ell + 1), sphere_dim=self.sphere_dim)))
        self.spherical_harmonics = SphericalHarmonics(max_ell=self.max_ell, sphere_dim=self.sphere_dim)

    @property 
    def ells(self):
        return jnp.arange(self.max_ell + 1)
    
    @jax.jit 
    def spectrum(self) -> Num[Array, "O L"]:
        return jax.vmap(
            lambda kappa, nu, variance: matern_spectrum(self.ells, kappa, nu, variance, dim=self.sphere_dim)
        )(self.kappa, self.nu, self.variance)
    
    @jax.jit 
    def from_spectrum(self, spectrum: Float[Array, "O L"], x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, "O"]:
        return jax.vmap(
            lambda spectrum: addition_theorem_scalar_kernel(
                spectrum, 
                sphere_addition_theorem(x, y, max_ell=self.max_ell, sphere_dim=self.sphere_dim)
            )
        )(spectrum)
    
    @jax.jit 
    def __call__(self, x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, "O"]:
        return self.from_spectrum(self.spectrum(), x, y)
    
    @jax.jit
    def phi(self, x: Float[Array, "N D"]) -> Float[Array, "N O F"]:
        spectrum = self.spectrum() # [O L]

        weights = jax.vmap(
            lambda s: jnp.repeat(
                jnp.sqrt(s),
                num_phases_in_frequency(frequency=self.ells, sphere_dim=self.sphere_dim),
                total_repeat_length=self.total_repeat_length,
            ), 
        )(spectrum) # [O F]
        phi = self.spherical_harmonics.polynomial_expansion(x) # [F N]

        return jnp.einsum("fn,of->nof", phi, weights)
        return weights * phi


### 
# HODGE MATERN
###

@jax.jit
def sph_dot_product(sph1: Float[Array, "N 2"], sph2: Float[Array, "N 2"]) -> Float[Array, "N"]:
    """
    Computes dot product in R^3 of two points on the sphere in spherical coordinates.
    """
    colat1, lon1 = sph1[..., 0], sph1[..., 1]
    colat2, lon2 = sph2[..., 0], sph2[..., 1]
    return jnp.sin(colat1) * jnp.sin(colat2) * jnp.cos(lon1 - lon2) + jnp.cos(colat1) * jnp.cos(colat2)


def sph_gegenbauer(x, y, max_ell: int, alpha: float = 0.5):
    return gegenbauer(x=sph_dot_product(x, y), max_ell=max_ell, alpha=alpha)


@Partial(jax.jit, static_argnames=("max_ell", "alpha"))
def hodge_gegenbauer(t1: Float[Array, "2"], t2: Float[Array, "2"], max_ell: int, alpha: float):
    Nt1 = tangent_basis_normalization_matrix(t1)
    Nt2 = tangent_basis_normalization_matrix(t2)
    dd = jax.jacfwd(jax.jacfwd(lambda x, y: sph_gegenbauer(x, y, max_ell=max_ell, alpha=alpha)[1:], argnums=0), argnums=1)(t1, t2)
    return Nt1.T @ dd @ Nt2
    

@Partial(jax.jit, static_argnames=('max_ell', ))
def hodge_sphere_addition_theorem(x: Float[Array, "2"], y: Float[Array, "2"], max_ell: int) -> Float[Array, "I 2 2"]:
    alpha = (2.0 - 1.0) / 2.0
    ells = jnp.arange(1, max_ell + 1, dtype=jnp.float64)
    gegenbauer_at_0 = 2 * ells + 1
    lambda_ells = ells * (ells + 1)
    ddPz = hodge_gegenbauer(x, y, max_ell, alpha)
    return (gegenbauer_at_0 / lambda_ells)[:, None, None] * ddPz


@jax.jit
def addition_theorem_matrix_kernel(spectrum: Float[Array, "I"], z: Float[Array, "I 2 2"]) -> Float[Array, "2 2"]:
    return jnp.sum(spectrum[:, None, None] * z, axis=0)


@dataclass 
class AbstractHodgeKernel(Module):
    nu: Float[Array, "1"] = param_field(jnp.array(2.5), bijector=tfb.Softplus())
    kappa: Float[Array, "1"] = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    variance: Float[Array, "1"] = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    alpha: float = static_field(0.5)
    max_ell: int = static_field(10)
    colatitude_min_value: float = static_field(1e-12) 
    sphere_dim: int = static_field(2)

    @property 
    def ells(self):
        return jnp.arange(1, self.max_ell + 1, dtype=jnp.float64)
    
    def spectrum(self) -> Float[Array, "I"]:
        return matern_spectrum(self.ells, self.kappa, self.nu, self.variance, dim=self.sphere_dim)
    
    @jax.jit
    def validate_inputs(self, x: Float[Array, "2"], y: Float[Array, "2"]) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        x = clip_colatitude(x, self.colatitude_min_value)
        y = clip_colatitude(y, self.colatitude_min_value)
        return x, y

    @jax.jit 
    def prepare_inputs(self, x: Float[Array, "2"], y: Float[Array, "2"]) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        x, y = self.validate_inputs(x, y)
        return hodge_sphere_addition_theorem(x, y, self.max_ell)
    

@dataclass
class HodgeMaternCurlFreeKernel(AbstractHodgeKernel):

    @jax.jit 
    def from_addition_theorem(self, z: Float[Array, "2 2"]) -> Float[Array, "2 2"]:
        return addition_theorem_matrix_kernel(self.spectrum(), z)

    def __call__(self, x: Float[Array, "2"], y: Float[Array, "2"]) -> Float[Array, "2 2"]:
        z = self.prepare_inputs(x, y)
        return self.from_addition_theorem(z)


@dataclass
class HodgeMaternDivFreeKernel(AbstractHodgeKernel):

    @jax.jit 
    def from_addition_theorem(self, z: Float[Array, "2 2"]) -> Float[Array, "2 2"]:
        dd = addition_theorem_matrix_kernel(self.spectrum(), z)
        H = hodge_star_matrix # [2 2]
        return H.T @ dd @ H

    def __call__(self, x: Float[Array, "2"], y: Float[Array, "2"]) -> Float[Array, "2 2"]:
        z = self.prepare_inputs(x, y)
        return self.from_addition_theorem(z)


@dataclass
class HodgeMaternKernel(Module):
    kappa: InitVar[ScalarFloat] = 1.0
    nu: InitVar[ScalarFloat] = 2.5
    variance: InitVar[ScalarFloat] = 1.0

    max_ell: int = static_field(10)
    curl_free_kernel: HodgeMaternCurlFreeKernel = param_field(init=False)
    div_free_kernel: HodgeMaternDivFreeKernel = param_field(init=False)
    colatitude_min_value: float = static_field(1e-12)

    def __post_init__(self, kappa, nu, variance):
        self.curl_free_kernel = HodgeMaternCurlFreeKernel(kappa=kappa, nu=nu, variance=variance, max_ell=self.max_ell, colatitude_min_value=self.colatitude_min_value)
        self.div_free_kernel = HodgeMaternDivFreeKernel(kappa=kappa, nu=nu, variance=variance, max_ell=self.max_ell, colatitude_min_value=self.colatitude_min_value)

    @jax.jit 
    def from_addition_theorem(self, z: Float[Array, "2 2"]) -> Float[Array, "2 2"]:
        return self.curl_free_kernel.from_addition_theorem(z) + self.div_free_kernel.from_addition_theorem(z)

    def spectrum(self):
        return jnp.concat([self.curl_free_kernel.spectrum(), self.div_free_kernel.spectrum()])

    def validate_inputs(self, x: Float[Array, "2"], y: Float[Array, "2"]) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        x = clip_colatitude(x, self.colatitude_min_value)
        y = clip_colatitude(y, self.colatitude_min_value)
        return x, y
    
    def __call__(self, x: Float[Array, "2"], y: Float[Array, "2"]) -> Float[Array, "2 2"]:
        x, y = self.validate_inputs(x, y)
        z = hodge_sphere_addition_theorem(x, y, self.max_ell)
        return self.from_addition_theorem(z)