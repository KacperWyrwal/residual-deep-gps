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
from dataclasses import dataclass
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from kernels import SphereMaternKernel, MultioutputSphereMaternKernel, EuclideanMaternKernel32, MultioutputEuclideanMaternKernel32, HodgeMaternKernel # TODO Change this to typevar or abstract class
from spherical_harmonics import SphericalHarmonics, SphericalHarmonicFields
from utils import sphere_to_tangent, sphere_expmap, jax_key_to_numpy_generator, num_phases_to_num_levels, expmap_sph, sph_to_car
from scipy.cluster.vq import kmeans



### 
# SHALLOW GPS
###


@dataclass 
class Prior(Module):
    kernel: SphereMaternKernel = param_field()
    jitter: Float = static_field(1e-12)


@dataclass 
class MultioutputPrior(Module):
    kernel: MultioutputSphereMaternKernel = param_field()
    jitter: Float = static_field(1e-12)

    @property 
    def num_outputs(self):
        return self.kernel.num_outputs
    

@dataclass 
class GaussianLikelihood(Module):
    noise_variance: Float = param_field(jnp.array(1.0), bijector=tfb.Softplus())


@jax.jit 
def exact_posterior_moments(
    Ktt: Float[Array, ""],
    Kxt: Float[Array, "N"],
    Kxx: Float[Array, "N N"],
    y: Float[Array, "N"],
    sigma_squared: Float[Array, ""],
    jitter: float = 1e-12,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    Kxx_plus_sigma_squared_I = Kxx.at[jnp.diag_indices_from(Kxx)].add(sigma_squared)
    Lxx = jnp.linalg.cholesky(Kxx_plus_sigma_squared_I)
    Lxx_inv_Kxt = jnp.linalg.solve(Lxx, Kxt) # [N N] @ [N] -> [N]
    Lxx_inv_y = jnp.linalg.solve(Lxx, y) # [N N] @ [N] -> [N]

    variance = Ktt - jnp.sum(jnp.square(Lxx_inv_Kxt), axis=0) # [] + ([N] -> []) -> []
    variance += jitter

    mean = jnp.sum(Lxx_inv_Kxt * Lxx_inv_y, axis=0)

    return mean, variance
    

@dataclass
class Posterior(Module):
    prior: Prior = param_field()
    likelihood: Module = param_field()

    @jax.jit 
    def moments(
        self, 
        t: Float[Array, "D"], 
        x: Float[Array, "N D"], 
        y: Float[Array, "N"]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        kernel = self.prior.kernel
        sigma_squared = self.likelihood.noise_variance
        jitter = self.prior.jitter

        Ktt = kernel(t, t)
        Kxt = jax.vmap(lambda x: kernel(x, t))(x)
        Kxx = jax.vmap(lambda x1: jax.vmap(lambda x2: kernel(x1, x2))(x))(x)

        return exact_posterior_moments(
            Ktt, Kxt, Kxx, y, sigma_squared, jitter=jitter
        )
    

@dataclass
class MultioutputPosterior(Module):
    prior: MultioutputPrior = param_field()
    likelihood: Module = param_field()

    @property 
    def num_outputs(self) -> int:
        return self.prior.num_outputs
    

@dataclass 
class HodgePrior(Module):
    kernel: HodgeMaternKernel = param_field()
    jitter: float = static_field(1e-12)
    

@dataclass 
class HodgePosterior(Module):
    prior: HodgePrior = param_field()
    likelihood: Module = param_field()


### 
# VARIATIONAL FAMILIES 
###

def kmeans_inducing_points(key: Key, x: Float[Array, "N D"], num_inducing: int) -> Float[Array, "M D"]:
    seed = jax_key_to_numpy_generator(key)

    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]

    k_centers = []
    while num_inducing > n:
        k_centers.append(kmeans(x, n)[0])
        num_inducing -= n
    k_centers.append(kmeans(x, num_inducing, seed=seed)[0])
    k_centers = np.concatenate(k_centers, axis=0)
    return jnp.asarray(k_centers, dtype=jnp.float64)


@Partial(jax.jit, static_argnames=('jitter',))
def inducing_points_moments(
    Kxx: Float[Array, ""], 
    Kzx: Float[Array, "M"], 
    Kzz: Float[Array, "M M"], 
    m: Float[Array, "M"], 
    sqrtS: Float[Array, "M M"], 
    jitter: float = 1e-12
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    Kzz = Kzz.at[jnp.diag_indices_from(Kzz)].add(jitter)

    Lzz = jnp.linalg.cholesky(Kzz)
    Lzz_inv_Kzx = jnp.linalg.solve(Lzz, Kzx) # [M M] @ [M] -> [M]
    sqrtS_T_Lzz_inv_Kzx = sqrtS.T @ Lzz_inv_Kzx # [M M] @ [M] -> [M]

    variance = (
        Kxx
        + jnp.sum(jnp.square(sqrtS_T_Lzz_inv_Kzx))
        - jnp.sum(jnp.square(Lzz_inv_Kzx))
    )
    variance += jitter

    mean = jnp.inner(Lzz_inv_Kzx, m) # [M] @ [M] -> []

    return mean, variance


@Partial(jax.jit, static_argnames=('jitter',))
def spherical_harmonic_features_moments(
    Kxx: Float[Array, ""], 
    Kxz: Float[Array, "M"], 
    Kzz_inv_diag: Float[Array, "M"], 
    m: Float[Array, "M"], 
    sqrtS: Float[Array, "M M"], 
    jitter: float = 1e-12
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    Lzz_T_inv_diag = jnp.sqrt(Kzz_inv_diag) / jnp.sqrt(1 + jitter * Kzz_inv_diag)
    Kxz_Lzz_T_inv = Kxz * Lzz_T_inv_diag
    Kxz_Lzz_T_inv_sqrtS = Kxz_Lzz_T_inv @ sqrtS

    covariance = (
        Kxx
        + jnp.sum(jnp.square(Kxz_Lzz_T_inv_sqrtS))
    )

    mean = (
        Kxz_Lzz_T_inv @ m
    )

    return mean, covariance


@jax.jit
def whitened_prior_kl(m: Float, sqrtS: Float) -> Float:
    S = sqrtS @ sqrtS.T
    qz = tfd.MultivariateNormalFullCovariance(loc=m, covariance_matrix=S)

    pz = tfd.MultivariateNormalFullCovariance(
        loc=jnp.zeros(m.shape), 
        covariance_matrix=jnp.eye(m.shape[0]),
    )
    return tfd.kl_divergence(qz, pz)


def inducing_points_prior_kl(m: Float, sqrtS: Float) -> Float:
    return whitened_prior_kl(m, sqrtS)


@jax.jit 
def sample_function__inducing_points(
    *, 
    phi: Callable[[Float[Array, "N D"]], Float[Array, "N F"]],
    Kzz: Float[Array, "M M"],
    u: Float[Array, "M"],
    phiz: Float[Array, "M F"],
    K_z: Callable[[Float[Array, "N D"]], Float[Array, "N M"]],
    key: Key,
) -> Callable[[Float[Array, "N D"]], Float[Array, "N"]]:
    
    # prior
    num_features = phiz.shape[1] # [F]
    w = jax.random.uniform(key, (num_features,)) # [F]

    # update Kzz^{-1}(u - phiz)
    # = (Lzz @ Lzz^T)^{-1}(u - phiz) 
    # = Lzz^{-T}Lzz^{-1}(u - phiz)
    # = Lzz^{-T}(u' - Lzz^{-1}phiz)
    Lzz = jnp.linalg.cholesky(Kzz) # [M M]
    fz = phiz @ w # [M F] @ [F] -> [M]
    Lzz_inv_fz = jnp.linalg.solve(Lzz, fz) # [M M] @ [M] -> [M]
    Kzz_inv_u_minus_fz = jnp.linalg.solve(Lzz.T, u - Lzz_inv_fz) # [M M] @ ([M] - [M]) -> [M]

    @jax.jit
    def f(x: Float[Array, "N D"]) -> Float[Array, "N"]:
        # prior 
        fx = phi(x) @ w # [N F] @ [F] -> [N]

        # update 
        Kxz = K_z(x) # [N M]
        update = Kxz @ Kzz_inv_u_minus_fz # [N M] @ [M] -> [N]

        return fx + update # [N] + [N] -> [N]

    return f


@jax.jit 
def pathwise_sample__inducing_points(
    *, 
    phix: Float[Array, "N F"],
    Kzz: Float[Array, "M M"],
    u: Float[Array, "M"],
    phiz: Float[Array, "M F"],
    Kxz: Float[Array, "N M"],
    key: Key,
) -> Float[Array, "N"]:
    
    # prior
    num_features = phiz.shape[1] # [F]
    w = jax.random.uniform(key, (num_features,)) # [F]
    fx = phix @ w # [N F] @ [F] -> [N]
    prior = fx

    # update Kzz^{-1}(u - phiz)
    # = (Lzz @ Lzz^T)^{-1}(u - phiz) 
    # = Lzz^{-T}Lzz^{-1}(u - phiz)
    # = Lzz^{-T}(u' - Lzz^{-1}phiz)

    Kzz = Kzz.at[jnp.diag_indices_from(Kzz)].add(1e-12)

    Lzz = jnp.linalg.cholesky(Kzz) # [M M]
    fz = phiz @ w # [M F] @ [F] -> [M]
    Lzz_inv_fz = jnp.linalg.solve(Lzz, fz) # [M M] @ [M] -> [M]
    Kzz_inv_u_minus_fz = jnp.linalg.solve(Lzz.T, u - Lzz_inv_fz) # [M M] @ ([M] - [M]) -> [M]
    update = Kxz @ Kzz_inv_u_minus_fz # [N M] @ [M] -> [N]

    return prior + update 

@dataclass 
class DummyPosterior(Module):
    prior: Prior = param_field()


@dataclass 
class MultioutputDummyPosterior(Module):
    prior: MultioutputPrior = param_field()

    @property 
    def num_outputs(self):
        return self.prior.num_outputs


@dataclass 
class InducingPointsPosterior(Module):
    posterior: Posterior = param_field()
    z: Float[Array, "M D"] = param_field(trainable=False)
    m: Float[Array, "M"] = param_field(init=False)
    sqrtS: Float[Array, "M M"] = param_field(init=False, bijector=tfb.FillTriangular())

    num_inducing: int = static_field(init=False)

    def __post_init__(self):
        self.num_inducing = self.z.shape[0]

        self.m = jnp.zeros(self.num_inducing)
        self.sqrtS = jnp.eye(self.num_inducing)

    @property 
    def jitter(self):
        return self.posterior.prior.jitter

    def prior_kl(self) -> Float:
        return inducing_points_prior_kl(self.m, self.sqrtS)

    @jax.jit 
    def moments(self, x: Float[Array, "D"]) -> tuple[Float[Array, "1"], Float[Array, "1"]]:
        kernel = self.posterior.prior.kernel
        z = self.z

        Kxx = kernel(x, x)
        Kzx = jax.vmap(lambda t: kernel(t, x))(z)
        Kzz = jax.vmap(lambda t1: jax.vmap(lambda t2: kernel(t1, t2))(z))(z)

        return inducing_points_moments(Kxx, Kzx, Kzz, self.m, self.sqrtS, jitter=self.jitter)

    @jax.jit
    def diag(self, x: Float[Array, "N D"]) -> tfd.Normal:
        mean, variance = jax.vmap(self.moments)(x)
        return tfd.Normal(loc=mean, scale=jnp.sqrt(variance))

    @jax.jit 
    def pathwise_sample(self, x: Float[Array, "D"], *, key: Key):
        kernel = self.posterior.prior.kernel
        z = self.z
        sqrtS = self.sqrtS
        m = self.m

        phi = kernel.phi 
        phix = jax.vmap(phi)(x) # [N F] -> [N F]
        phiz = jax.vmap(phi)(z) # [M F] -> [M F]
        Kxz = jax.vmap(lambda t1: jax.vmap(lambda t2: kernel(t1, t2))(z))(x) # [N M]
        Kzz = jax.vmap(lambda t1: jax.vmap(lambda t2: kernel(t1, t2))(z))(z) # [M M]
        
        # sample from the variational distribution
        eps = jax.random.normal(key, (self.num_inducing,))
        u = m + sqrtS @ eps

        return pathwise_sample__inducing_points(
            phix=phix, Kzz=Kzz, u=u, phiz=phiz, Kxz=Kxz, key=key
        )
    


@dataclass 
class MultioutputInducingPointsPosterior(Module):
    posterior: MultioutputPosterior = param_field()
    z: Float[Array, "M D"] = param_field(trainable=False)
    m: Float[Array, "M O"] = param_field(init=False)
    sqrtS: Float[Array, "M M O"] = param_field(init=False, bijector=tfb.FillTriangular())

    num_outputs: int = static_field(init=False)
    num_inducing: int = static_field(init=False)

    def __post_init__(self):
        self.num_outputs = self.posterior.num_outputs
        self.num_inducing = self.z.shape[0]

        self.m = jnp.zeros((self.num_outputs, self.num_inducing))
        self.sqrtS = jnp.repeat(jnp.expand_dims(jnp.eye(self.num_inducing), axis=0), self.num_outputs, axis=0)

    @jax.jit 
    def prior_kl(self) -> Float:
        return jnp.sum(jax.vmap(inducing_points_prior_kl)(self.m, self.sqrtS), axis=0)

    @jax.jit 
    def moments(self, x: Float[Array, "D"]) -> tuple[Float[Array, "O 1"], Float[Array, "O 1"]]:
        kernel = self.posterior.prior.kernel 
        z = self.z

        Kxx = kernel(x, x)
        Kzx = jax.vmap(lambda t: kernel(t, x))(z)
        Kzz = jax.vmap(lambda t1: jax.vmap(lambda t2: kernel(t1, t2))(z))(z)

        mean, variance = jax.vmap(inducing_points_moments, in_axes=(0, 1, 2, 0, 0))(Kxx, Kzx, Kzz, self.m, self.sqrtS)

        return mean, variance

    @jax.jit 
    def diag(self, x: Float[Array, "N D"]) -> tfd.Normal:
        mean, variance = jax.vmap(self.moments)(x)
        return tfd.Normal(loc=mean, scale=jnp.sqrt(variance))
    
    @jax.jit
    def pathwise_sample(self, x: Float[Array, "N D"], *, key: Key) -> Float[Array, "N O"]:
        kernel = self.posterior.prior.kernel
        z = self.z
        sqrtS = self.sqrtS # [O M M]
        m = self.m # [O M]

        phi = kernel.phi 
        phix = phi(x).mT
        phiz = phi(z).mT
        # phix = jax.vmap(phi)(x).mT # [N O F] -> [N F O]
        # phiz = jax.vmap(phi)(z).mT # [M O F] -> [M F O]
        Kxz = jax.vmap(lambda t1: jax.vmap(lambda t2: kernel(t1, t2))(z))(x) # [N M O]
        Kzz = jax.vmap(lambda t1: jax.vmap(lambda t2: kernel(t1, t2))(z))(z) # [M M O]
        
        # sample from the variational distribution
        eps = jax.random.normal(key, (self.num_outputs, self.num_inducing))
        u = m + jnp.einsum("onm, om -> om", sqrtS, eps)

        res = jax.vmap(
            lambda phix, Kzz, u, phiz, Kxz: pathwise_sample__inducing_points(
                phix=phix, Kzz=Kzz, u=u, phiz=phiz, Kxz=Kxz, key=key
            ), 
            in_axes=(2, 2, 0, 2, 2), 
            out_axes=1,
        )(phix, Kzz, u, phiz, Kxz)

        return res
    

@dataclass
class SphericalHarmonicFeaturesPosterior(Module):
    posterior: Posterior = param_field()
    spherical_harmonics: SphericalHarmonics = static_field()
    m: Float[Array, "M"] = param_field(init=False)
    sqrtS: Float[Array, "M M"] = param_field(init=False, bijector=tfb.FillTriangular())
    sqrtS_augment: Float[Array, "L"] = param_field(init=False)
    num_inducing: int = static_field(init=False)

    def __post_init__(self):
        kernel = self.posterior.prior.kernel

        self.num_inducing = self.spherical_harmonics.num_phases
        self.m = jnp.zeros(self.num_inducing)
        self.sqrtS = jnp.eye(self.num_inducing)
        self.sqrtS_augment = jnp.ones(kernel.max_ell + 1).at[:self.spherical_harmonics.max_ell + 1].set(0.0)

    @jax.jit 
    def Kzz_diag(self, spectrum: Float[Array, "L"]) -> Float[Array, "M"]:
        shf = self.spherical_harmonics
        repeats = np.array(shf.num_phases_per_frequency)
        total_repeat_length = shf.num_phases
        return jnp.repeat(
            spectrum[:shf.max_ell + 1], 
            repeats=repeats,
            total_repeat_length=total_repeat_length,
        )
    
    def Kxz(self, x: Float[Array, "D"]) -> Float[Array, "M"]:
        return self.spherical_harmonics.polynomial_expansion(x).T
    
    def prior_kl(self) -> Float[Array, ""]:
        return whitened_prior_kl(self.m, self.sqrtS)

    @jax.jit
    def moments(self, x: Float[Array, "N D"]) -> tuple[Float[Array, ""], Float[Array, ""]]:
        kernel = self.posterior.prior.kernel

        spectrum = kernel.spectrum()

        # This already accounts for the subtraction of the identity matrix from S'
        S_augment = jnp.square(self.sqrtS_augment)
        Kxx = kernel.from_spectrum(spectrum * S_augment, x, x)
        Kzz_diag = self.Kzz_diag(spectrum)
        Kxz = self.Kxz(x)

        return spherical_harmonic_features_moments(Kxx, Kxz, Kzz_diag, self.m, self.sqrtS)
    
    @jax.jit 
    def diag(self, x: Float[Array, "N D"]) -> tfd.Normal:
        mean, variance = jax.vmap(self.moments)(x)
        return tfd.Normal(loc=mean, scale=jnp.sqrt(variance))


@dataclass
class MultioutputSphericalHarmonicFeaturesPosterior(Module):
    num_outputs: int = static_field(init=False)

    posterior: MultioutputPosterior = param_field()
    spherical_harmonics: SphericalHarmonics = static_field()
    m: Float[Array, "M"] = param_field(init=False)
    sqrtS: Float[Array, "M M"] = param_field(init=False, bijector=tfb.FillTriangular())
    sqrtS_augment: Float[Array, "L"] = param_field(init=False)

    def __post_init__(self):
        kernel = self.posterior.prior.kernel

        self.num_outputs = self.posterior.num_outputs
        
        num_inducing = self.spherical_harmonics.num_phases
        self.m = jnp.zeros(num_inducing)
        self.sqrtS = jnp.eye(num_inducing)
        self.sqrtS_augment = jnp.ones(kernel.max_ell + 1).at[:self.spherical_harmonics.max_ell + 1].set(0.0)

        self.m = jnp.broadcast_to(self.m, (self.num_outputs, num_inducing))
        self.sqrtS = jnp.broadcast_to(self.sqrtS, (self.num_outputs, num_inducing, num_inducing))
        self.sqrtS_augment = jnp.broadcast_to(self.sqrtS_augment, (self.num_outputs, kernel.max_ell + 1))

    @jax.jit
    def prior_kl(self) -> Float:
        return jnp.sum(jax.vmap(whitened_prior_kl)(self.m, self.sqrtS), axis=0)

    @jax.jit 
    def Kzz_diag(self, spectrum: Float[Array, "O L"]) -> Float[Array, "O M"]:
        shf = self.spherical_harmonics
        repeats = np.array(shf.num_phases_per_frequency)
        total_repeat_length = shf.num_phases
        return jax.vmap(
            lambda spectrum: jnp.repeat(spectrum, repeats=repeats, total_repeat_length=total_repeat_length)
        )(spectrum[:, :shf.max_ell + 1])
    

    def Kxz(self, x: Float[Array, "D"]) -> Float[Array, "O M"]:
        return self.spherical_harmonics.polynomial_expansion(x).T
    
    
    @jax.jit
    def moments(self, x: Float[Array, "D"]) -> tuple[Float[Array, "O"], Float[Array, "O"]]:
        kernel = self.posterior.prior.kernel

        # prior covariance adjusted by the diagonal variational parameters 
        spectrum = kernel.spectrum() # [O L]
        S_augment = jnp.square(self.sqrtS_augment) # [O L]
        Kxx = kernel.from_spectrum(spectrum * S_augment, x, x) # [O N N]

        # variational covariance 
        Kzz_diag = self.Kzz_diag(spectrum) # [O M]
        Kxz = self.Kxz(x) # [O M]

        m = self.m
        sqrtS = self.sqrtS

        return jax.vmap(
            lambda Kxx, Kzz_diag, m, sqrtS: spherical_harmonic_features_moments(Kxx, Kxz, Kzz_diag, m, sqrtS)
        )(Kxx, Kzz_diag, m, sqrtS)
    
    @jax.jit 
    def diag(self, x: Float[Array, "N D"]) -> tfd.Normal:
        mean, variance = jax.vmap(self.moments)(x)
        return tfd.Normal(loc=mean, scale=jnp.sqrt(variance))


@Partial(jax.jit, static_argnames=("jitter", ))
def spherical_harmonic_fields_moments(
    Kxz: Float[Array, "2 I"], Kzz_diag: Float[Array, "I"], m: Float[Array, "2"], sqrtS: Float[Array, "I I"], jitter: float
) -> tuple[Float[Array, "I"], Float[Array, "I"]]:
    # Kxz @ K
    Lzz_T_inv = jnp.sqrt(Kzz_diag / (1 + Kzz_diag * jitter))

    Kxz_Lzz_T_inv = Kxz * Lzz_T_inv 
    Kxz_Lzz_T_inv_sqrtS = Kxz_Lzz_T_inv @ sqrtS

    covariance = Kxz_Lzz_T_inv_sqrtS @ Kxz_Lzz_T_inv_sqrtS.T
    covariance = covariance.at[jnp.diag_indices_from(covariance)].add(jitter)

    mean = Kxz_Lzz_T_inv @ m
    return mean, covariance

    
@dataclass 
class SphericalHarmonicFieldsPosterior(Module):
    posterior: HodgePosterior = param_field()
    spherical_harmonic_fields: SphericalHarmonicFields = param_field()
    m: Float[Array, "M"] = param_field(init=False)
    sqrtS: Float[Array, "M M"] = param_field(init=False, bijector=tfb.FillTriangular())

    def __post_init__(self):
        self.m = jnp.zeros(self.num_inducing)
        self.sqrtS = jnp.eye(self.num_inducing)

    @property 
    def jitter(self):
        return self.posterior.prior.jitter

    @property 
    def kernel(self):
        return self.posterior.prior.kernel 

    @property 
    def num_inducing(self):
        return self.spherical_harmonic_fields.num_fields

    def prior_kl(self) -> Float:
        return whitened_prior_kl(self.m, self.sqrtS)
    
    # FIXME This is the main ugly part of the code. It would be nice to refactor, but it's not a priority.
    @jax.jit 
    def Kzz_diag(self) -> Float[Array, "I"]:
        curl_free_kernel = self.kernel.curl_free_kernel
        div_free_kernel = self.kernel.div_free_kernel

        curl_free_ahats_per_frequency = curl_free_kernel.spectrum()[:self.spherical_harmonic_fields.max_ell]
        div_free_ahats_per_frequency = div_free_kernel.spectrum()[:self.spherical_harmonic_fields.max_ell]

        def repeat_per_phase(x):
            return jnp.repeat(
                x, 
                self.spherical_harmonic_fields.num_phases_per_frequency,
                total_repeat_length=self.spherical_harmonic_fields.num_phases,
            )

        return jnp.concatenate([
            repeat_per_phase(curl_free_ahats_per_frequency),
            repeat_per_phase(div_free_ahats_per_frequency),
        ]) 
    
    def Kxz(self, x: Float[Array, "2"]) -> Float[Array, "2 I"]:
        return self.spherical_harmonic_fields(x).T 

    @jax.jit 
    def moments(self, x: Float[Array, "2"]) -> tuple[Float[Array, "1"], Float[Array, "1"]]:
        mean, covariance = spherical_harmonic_fields_moments(
            self.Kxz(x), self.Kzz_diag(), self.m, self.sqrtS, jitter=self.jitter
        )
        return mean, covariance
    
    @jax.jit 
    def diag(self, x: Float[Array, "N 2"]) -> tfd.MultivariateNormalFullCovariance:
        mean, covariance = jax.vmap(self.moments)(x)
        return tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=covariance)
    

### 
# DEEP GPS
### 


@dataclass 
class SphereResidualDeepGP(Module):
    hidden_layers: list[MultioutputInducingPointsPosterior] = param_field()
    output_layer: InducingPointsPosterior = param_field()
    num_samples: int = static_field(1)

    @property 
    def posterior(self) -> Posterior:
        return self.output_layer.posterior      
    
    def prior_kl(self) -> Float:
        return sum(layer.prior_kl() for layer in self.hidden_layers) + self.output_layer.prior_kl()
    
    def sample_moments(self, x: Float[Array, "N D"], *, key: Key) -> tfd.Normal:
        hidden_layer_keys = jax.random.split(key, len(self.hidden_layers))
        for hidden_layer_key, layer in zip(hidden_layer_keys, self.hidden_layers):
            v = layer.diag(x).sample(seed=hidden_layer_key)
            u = sphere_to_tangent(x, v)
            x = sphere_expmap(x, u)
        return jax.vmap(self.output_layer.moments)(x)

    def diag(self, x: Float[Array, "N D"], *, key: Key) -> tfd.MixtureSameFamily:
        sample_keys = jax.random.split(key, self.num_samples)

        # In MixtureSameFamily batch size goes last; hence, out_axes = 1
        mean, variance = jax.vmap(lambda k: self.sample_moments(x, key=k), out_axes=1)(sample_keys) 

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=jnp.zeros(self.num_samples)), 
            components_distribution=tfd.Normal(loc=mean, scale=jnp.sqrt(variance)), 
        )


@dataclass
class DeepGaussianLikelihood(Module):
    noise_variance: Float = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    
    @jax.jit 
    def diag(self, pf: tfd.MixtureSameFamily) -> tfd.MixtureSameFamily:
        component_distribution = pf.components_distribution
        mean, variance = component_distribution.mean(), component_distribution.variance()
        variance += self.noise_variance
        return tfd.MixtureSameFamily(
            mixture_distribution=pf.mixture_distribution,
            components_distribution=tfd.Normal(loc=mean, scale=jnp.sqrt(variance)),
        )


@dataclass 
class EuclideanDeepGP(Module):
    hidden_layers: list[InducingPointsPosterior] = param_field()
    output_layer: InducingPointsPosterior = param_field()
    num_samples: int = static_field(1)

    @property 
    def posterior(self) -> MultioutputPosterior:
        return self.output_layer.posterior        
    
    def prior_kl(self) -> Float:
        return sum(layer.prior_kl() for layer in self.hidden_layers) + self.output_layer.prior_kl()

    def sample_moments(self, x: Float[Array, "N D"], *, key: Key) -> tfd.Normal:
        hidden_layer_keys = jax.random.split(key, len(self.hidden_layers))
        for hidden_layer_key, layer in zip(hidden_layer_keys, self.hidden_layers):
            v = layer.diag(x).sample(seed=hidden_layer_key) # [N D]
            x = x + v # euclidean expmap
        return jax.vmap(self.output_layer.moments)(x)
    
    def diag(self, x: Float[Array, "N D"], *, key: Key) -> tfd.MixtureSameFamily:
        sample_keys = jax.random.split(key, self.num_samples)

        # In MixtureSameFamily batch size goes last; hence, out_axes = 1
        mean, variance = jax.vmap(lambda k: self.sample_moments(x, key=k), out_axes=1)(sample_keys) 

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=jnp.zeros(self.num_samples)), 
            components_distribution=tfd.Normal(loc=mean, scale=jnp.sqrt(variance)), 
        )
    

@dataclass 
class HodgeResidualDeepGP(Module):
    hidden_layers: list[SphericalHarmonicFieldsPosterior] = param_field()
    output_layer: SphericalHarmonicFeaturesPosterior = param_field()
    num_samples: int = static_field(1)

    @property 
    def posterior(self):
        return self.output_layer.posterior
    
    def sample_moments(self, x: Float[Array, "N 2"], *, key: Key) -> tfd.Normal:
        hidden_layer_keys = jax.random.split(key, len(self.hidden_layers))
        for hidden_layer_key, layer in zip(hidden_layer_keys, self.hidden_layers):
            v = layer.diag(x).sample(seed=hidden_layer_key)
            x = jax.vmap(expmap_sph, in_axes=(0, 0))(x, v)
        return jax.vmap(self.output_layer.moments)(sph_to_car(x))

    def diag(self, x: Float[Array, "N 2"], *, key: Key) -> tfd.MixtureSameFamily:
        sample_keys = jax.random.split(key, self.num_samples)

        # In MixtureSameFamily batch size goes last; hence, out_axes = 1
        mean, variance = jax.vmap(lambda k: self.sample_moments(x, key=k), out_axes=1)(sample_keys)

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=jnp.zeros(self.num_samples)), 
            components_distribution=tfd.Normal(loc=mean, scale=jnp.sqrt(variance)), 
        )
    
    def prior_kl(self) -> Float:
        return sum(layer.prior_kl() for layer in self.hidden_layers) + self.output_layer.prior_kl()


def create_residual_deep_gp_with_spherical_harmonic_features(
    num_layers: int, total_hidden_variance: float, num_inducing: int, x: Float[Array, "N D"], num_samples: int = 3, *, key: Key, 
    nu: float = 1.5, kernel_max_ell: int | None = None
) -> SphereResidualDeepGP:
    sphere_dim = x.shape[1] - 1

    hidden_nu = jnp.array(nu)
    output_nu = hidden_nu

    hidden_variance = jnp.array(total_hidden_variance / max(num_layers - 1, 1))
    output_variance = jnp.array(1.0)

    hidden_kappa = jnp.array(1.0)
    output_kappa = hidden_kappa

    shf_max_ell = num_phases_to_num_levels(num_inducing, sphere_dim=sphere_dim)
    if kernel_max_ell is None:
        kernel_max_ell = shf_max_ell
    hidden_spherical_harmonics = SphericalHarmonics(max_ell=shf_max_ell, sphere_dim=sphere_dim)
    output_spherical_harmonics = hidden_spherical_harmonics

    hidden_layers = []
    for _ in range(num_layers - 1):
        kernel = MultioutputSphereMaternKernel(
            num_outputs=sphere_dim + 1, 
            sphere_dim=sphere_dim, 
            nu=hidden_nu,
            kappa=hidden_kappa,
            variance=hidden_variance,
            max_ell=kernel_max_ell,
        )
        prior = MultioutputPrior(kernel=kernel)
        posterior = MultioutputDummyPosterior(prior=prior)
        layer = MultioutputSphericalHarmonicFeaturesPosterior(posterior=posterior, spherical_harmonics=hidden_spherical_harmonics)
        hidden_layers.append(layer)

    kernel = SphereMaternKernel(
        sphere_dim=sphere_dim,
        nu=output_nu,
        kappa=output_kappa,
        variance=output_variance,
        max_ell=kernel_max_ell,
    )
    prior = Prior(kernel=kernel)
    likelihood = DeepGaussianLikelihood()
    posterior = Posterior(prior=prior, likelihood=likelihood)
    output_layer = SphericalHarmonicFeaturesPosterior(posterior=posterior, spherical_harmonics=output_spherical_harmonics)

    return SphereResidualDeepGP(hidden_layers=hidden_layers, output_layer=output_layer, num_samples=num_samples)


def create_euclidean_deep_gp_with_inducing_points(
    num_layers: int, total_hidden_variance: float, num_inducing: int, x: Float[Array, "N D"], num_samples: int = 3, *, key: Key, train_inducing: bool = True
) -> EuclideanDeepGP:
    sphere_dim = x.shape[1] - 1
    
    hidden_variance = total_hidden_variance / max(num_layers - 1, 1)
    output_variance = jnp.array(1.0)

    hidden_kappa = jnp.ones(sphere_dim + 1)
    output_kappa = hidden_kappa

    z = kmeans_inducing_points(key, x, num_inducing)
    hidden_z = z
    output_z = z

    hidden_layers = []
    for _ in range(num_layers - 1):
        kernel = MultioutputEuclideanMaternKernel32(
            variance=hidden_variance,
            kappa=hidden_kappa, 
            num_outputs=sphere_dim + 1, 
            num_inputs=sphere_dim + 1,
        )
        prior = MultioutputPrior(kernel=kernel)
        posterior = MultioutputDummyPosterior(prior=prior)
        layer = MultioutputInducingPointsPosterior(posterior=posterior, z=hidden_z)
        if train_inducing:
            layer = layer.replace_trainable(z=True)
        hidden_layers.append(layer)


    kernel = EuclideanMaternKernel32(
        num_inputs=sphere_dim + 1,
        variance=output_variance,
        kappa=output_kappa,
    )
    prior = Prior(kernel=kernel)
    likelihood = DeepGaussianLikelihood()
    posterior = Posterior(prior=prior, likelihood=likelihood)
    output_layer = InducingPointsPosterior(posterior=posterior, z=output_z)
    if train_inducing:
        output_layer = output_layer.replace_trainable(z=True)

    return EuclideanDeepGP(hidden_layers=hidden_layers, output_layer=output_layer, num_samples=num_samples)


def create_euclidean_deep_gp_with_input_geometric_layer_and_inducing_points(
    num_layers: int, total_hidden_variance: float, num_inducing: int, x: Float[Array, "N D"], num_samples: int = 3, *, key: Key, train_inducing: bool = True, 
) -> EuclideanDeepGP:
    if num_layers == 1:
        return create_residual_deep_gp_with_inducing_points(
            num_layers, total_hidden_variance, num_inducing, x, num_samples=num_samples, key=key,
        )
    
    sphere_dim = x.shape[1] - 1

    input_nu = jnp.array(1.5)

    input_variance = total_hidden_variance / max(num_layers - 1, 1)
    hidden_variance = input_variance
    output_variance = jnp.array(1.0)

    input_kappa = jnp.array(1.0)
    hidden_kappa = jnp.ones(sphere_dim + 1)
    output_kappa = hidden_kappa

    z = kmeans_inducing_points(key, x, num_inducing)
    input_z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    hidden_z = z
    output_z = z


    hidden_layers = []

    # input layer
    kernel = MultioutputSphereMaternKernel(
        num_outputs=sphere_dim + 1, 
        sphere_dim=sphere_dim, 
        variance=input_variance,
        kappa=input_kappa,
        nu=input_nu,
    )
    prior = MultioutputPrior(kernel=kernel)
    posterior = MultioutputDummyPosterior(prior=prior)
    layer = MultioutputInducingPointsPosterior(posterior=posterior, z=input_z)
    hidden_layers.append(layer)

    for _ in range(num_layers - 2):
        kernel = MultioutputEuclideanMaternKernel32(
            kappa=hidden_kappa,
            variance=hidden_variance, 
            num_inputs=sphere_dim + 1,
            num_outputs=sphere_dim + 1,
        )
        prior = MultioutputPrior(kernel=kernel)
        posterior = MultioutputDummyPosterior(prior=prior)
        layer = MultioutputInducingPointsPosterior(posterior=posterior, z=hidden_z)
        if train_inducing:
            layer = layer.replace_trainable(z=True)
        hidden_layers.append(layer)

    kernel = EuclideanMaternKernel32(
        kappa=output_kappa,
        variance=output_variance,
        num_inputs=sphere_dim + 1,
    )
    prior = Prior(kernel=kernel)
    likelihood = DeepGaussianLikelihood()
    # likelihood = GaussianLikelihood()
    posterior = Posterior(prior=prior, likelihood=likelihood)
    output_layer = InducingPointsPosterior(posterior=posterior, z=output_z)
    if train_inducing:
        output_layer = output_layer.replace_trainable(z=True)

    return EuclideanDeepGP(hidden_layers=hidden_layers, output_layer=output_layer, num_samples=num_samples)


def create_residual_deep_gp_with_inducing_points(
    num_layers: int, total_hidden_variance: float, num_inducing: int, x: Float[Array, "N D"], num_samples: int = 3, *, key: Key, train_inducing: bool = False, 
    nu: float = 1.5, kernel_max_ell: int | None = None
) -> SphereResidualDeepGP:
    sphere_dim = x.shape[1] - 1

    hidden_nu = jnp.array(nu)
    output_nu = hidden_nu

    hidden_variance = total_hidden_variance / max(num_layers - 1, 1)
    output_variance = jnp.array(1.0)

    hidden_kappa = jnp.array(1.0)
    output_kappa = hidden_kappa

    z = kmeans_inducing_points(key, x, num_inducing)
    hidden_z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    output_z = hidden_z

    if kernel_max_ell is None:
        kernel_max_ell = num_phases_to_num_levels(num_inducing, sphere_dim=sphere_dim)

    hidden_layers = []
    for _ in range(num_layers - 1):
        kernel = MultioutputSphereMaternKernel(
            num_outputs=sphere_dim + 1, 
            sphere_dim=sphere_dim, 
            nu=hidden_nu,
            kappa=hidden_kappa,
            variance=hidden_variance,
            max_ell=kernel_max_ell,
        )
        prior = MultioutputPrior(kernel=kernel)
        posterior = MultioutputDummyPosterior(prior=prior)
        layer = MultioutputInducingPointsPosterior(posterior=posterior, z=hidden_z) # TODO set z to be trainable maybe 
        hidden_layers.append(layer)

    kernel = SphereMaternKernel(
        sphere_dim=sphere_dim,
        nu=output_nu,
        kappa=output_kappa,
        variance=output_variance,
        max_ell=kernel_max_ell,
    )
    prior = Prior(kernel=kernel)
    likelihood = DeepGaussianLikelihood()
    # likelihood = GaussianLikelihood()
    posterior = Posterior(prior=prior, likelihood=likelihood)
    output_layer = InducingPointsPosterior(posterior=posterior, z=output_z)
    if train_inducing:
        output_layer = output_layer.replace_trainable(z=True)

    return SphereResidualDeepGP(hidden_layers=hidden_layers, output_layer=output_layer, num_samples=num_samples)


def create_hodge_residual_deep_gp_with_spherical_harmonic_features(
    num_layers: int, total_hidden_variance: float, num_inducing: int, x: Float[Array, "N D"], num_samples: int = 3, *, key: Key, 
    nu: float = 1.5, kernel_max_ell: int | None = None
) -> SphereResidualDeepGP:
    sphere_dim = x.shape[1]

    hidden_nu = jnp.array(nu)
    output_nu = hidden_nu

    hidden_variance = total_hidden_variance / max(num_layers - 1, 1)
    output_variance = jnp.array(1.0)

    hidden_kappa = jnp.array(1.0)
    output_kappa = hidden_kappa

    shf_hidden_max_ell = num_phases_to_num_levels(num_inducing // 2, sphere_dim=sphere_dim)
    # shf_output_max_ell = num_phases_to_num_levels(num_inducing, sphere_dim=sphere_dim) TODO temporary fix 
    shf_output_max_ell = num_phases_to_num_levels(49, sphere_dim=sphere_dim)
    if kernel_max_ell is None:
        hidden_kernel_max_ell = shf_hidden_max_ell
        output_kernel_max_ell = shf_output_max_ell
    hidden_spherical_harmonic_fields = SphericalHarmonicFields(max_ell=shf_hidden_max_ell)
    output_spherical_harmonics = SphericalHarmonics(max_ell=shf_output_max_ell, sphere_dim=sphere_dim)

    hidden_layers = []
    for _ in range(num_layers - 1):
        kernel = HodgeMaternKernel(
            kappa=hidden_kappa,
            nu=hidden_nu,
            variance=hidden_variance, 
            max_ell=hidden_kernel_max_ell,
        )
        prior = Prior(kernel=kernel)
        posterior = DummyPosterior(prior=prior)
        layer = SphericalHarmonicFieldsPosterior(posterior=posterior, spherical_harmonic_fields=hidden_spherical_harmonic_fields)
        hidden_layers.append(layer)

    kernel = SphereMaternKernel(
        sphere_dim=sphere_dim,
        nu=output_nu,
        kappa=output_kappa,
        variance=output_variance,
        max_ell=output_kernel_max_ell,
    )
    prior = Prior(kernel=kernel)
    likelihood = DeepGaussianLikelihood()
    posterior = Posterior(prior=prior, likelihood=likelihood)
    output_layer = SphericalHarmonicFeaturesPosterior(posterior=posterior, spherical_harmonics=output_spherical_harmonics)

    return HodgeResidualDeepGP(hidden_layers=hidden_layers, output_layer=output_layer, num_samples=num_samples)


def create_model(
    num_layers: int, total_hidden_variance: float, num_inducing: int, x: Float[Array, "N D"], num_samples: int = 3, *, key: Key, name: str, nu: float = 1.5, 
    kernel_max_ell: int | None = None
): 
    if name == 'euclidean+inducing_points':
        return create_euclidean_deep_gp_with_inducing_points(
            num_layers, total_hidden_variance, num_inducing, x, num_samples=num_samples, key=key,
        )
    if name == 'residual+spherical_harmonic_features':
        return create_residual_deep_gp_with_spherical_harmonic_features(
            num_layers, total_hidden_variance, num_inducing, x, num_samples=num_samples, key=key, nu=nu, kernel_max_ell=kernel_max_ell
        )
    if name == 'euclidean_with_geometric_input+inducing_points':
        return create_euclidean_deep_gp_with_input_geometric_layer_and_inducing_points(
            num_layers, total_hidden_variance, num_inducing, x, num_samples=num_samples, key=key,
        )
    if name == 'residual+inducing_points':
        return create_residual_deep_gp_with_inducing_points(
            num_layers, total_hidden_variance, num_inducing, x, num_samples=num_samples, key=key, nu=nu, kernel_max_ell=kernel_max_ell
        ) 
    if name == 'residual+hodge+spherical_harmonic_features':
        return create_hodge_residual_deep_gp_with_spherical_harmonic_features(
            num_layers, total_hidden_variance, num_inducing, x, num_samples=num_samples, key=key, nu=nu, kernel_max_ell=kernel_max_ell
        )
    raise ValueError(f"Unknown model name: {name}")


### 
# ELBO 
###

@jax.jit 
def expected_log_likelihood(y: Float, m: Float, f_var: Float, eps_var: Float) -> Float:
    log2pi = jnp.log(2 * jnp.pi)
    squared_error = jnp.square(y - m)
    return -0.5 * jnp.sum(log2pi + jnp.log(eps_var) + (squared_error + f_var) / eps_var, axis=-1)


@Partial(jax.jit, static_argnames=('n',))
def deep_negative_elbo(p, x: Float, y: Float, *, key: Key, n: int) -> Float:
    eps_var = p.posterior.likelihood.noise_variance
    sample_keys = jax.random.split(key, p.num_samples)

    def sample_expected_log_likelihood(key: Key) -> Float:
        m, f_var = p.sample_moments(x, key=key)
        return expected_log_likelihood(y, m, f_var, eps_var)
    
    deep_expected_log_likelihood = jnp.mean(jax.vmap(sample_expected_log_likelihood)(sample_keys), axis=0)
    batch_ratio_correction = n / x.shape[0]

    return -(deep_expected_log_likelihood * batch_ratio_correction - p.prior_kl())
