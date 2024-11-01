from jax import config 
config.update("jax_enable_x64", True)


import jax 
import jax.numpy as jnp
import gpjax 
import optax 
import tensorflow_probability.substrates.jax.distributions as tfd
from jaxtyping import Key, Array
import pandas as pd 
import numpy as np
from jaxtyping import Key 
import jax 
import jax.numpy as jnp
import numpy as np
import netCDF4
from jaxtyping import Array
from kernels import HodgeMaternKernel
from models import HodgeResidualDeepGP, DummyPosterior, Prior, Posterior, SphericalHarmonicFieldsPosterior
from spherical_harmonics import SphericalHarmonicFields
from gpjax.typing import Float 
from gpjax.base import Module, param_field, static_field
import tensorflow_probability.substrates.jax.bijectors as tfb
from dataclasses import dataclass 
from jax.tree_util import Partial 
from utils import expmap_sph, sph_to_car, sphere_uniform_grid


@dataclass 
class HodgeDeepGaussianLikelihood(Module):
    noise_variance: Float = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    
    @jax.jit 
    def diag(self, pf: tfd.MixtureSameFamily) -> tfd.MixtureSameFamily:
        component_distribution = pf.components_distribution
        mean, covariance = component_distribution.mean(), component_distribution.covariance()
        covariance += jnp.eye(mean.shape[-1]) * self.noise_variance
        return tfd.MixtureSameFamily(
            mixture_distribution=pf.mixture_distribution,
            components_distribution=tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=covariance),
        )


@dataclass 
class VectorOutputHodgeResidualDeepGP(Module):
    hidden_layers: list[SphericalHarmonicFieldsPosterior] = param_field()
    output_layer: SphericalHarmonicFieldsPosterior = param_field()
    num_samples: int = static_field(1)

    @property 
    def posterior(self):
        return self.output_layer.posterior
    
    def sample_moments(self, x: Float[Array, "N 2"], *, key: Key) -> tfd.Normal:
        hidden_layer_keys = jax.random.split(key, len(self.hidden_layers))
        for hidden_layer_key, layer in zip(hidden_layer_keys, self.hidden_layers):
            v = layer.diag(x).sample(seed=hidden_layer_key)
            x = jax.vmap(expmap_sph, in_axes=(0, 0))(x, v)
        return jax.vmap(self.output_layer.moments)(x)

    def diag(self, x: Float[Array, "N 2"], *, key: Key) -> tfd.MixtureSameFamily:
        sample_keys = jax.random.split(key, self.num_samples)

        # In MixtureSameFamily batch size goes last; hence, out_axes = 1
        mean, covariance = jax.vmap(lambda k: self.sample_moments(x, key=k), out_axes=1)(sample_keys) 

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=jnp.zeros(self.num_samples)), 
            components_distribution=tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=covariance),
        )
    
    def prior_kl(self) -> Float:
        return sum(layer.prior_kl() for layer in self.hidden_layers) + self.output_layer.prior_kl()



def create_hodge_residual_deep_gp_with_spherical_harmonic_features(
    num_layers: int, total_hidden_variance: float, max_ell: int, x: Float[Array, "N D"], num_samples: int = 3, *, key: Key, 
    nu: float = 1.5, kernel_max_ell: int | None = None
) -> HodgeResidualDeepGP:
    hidden_nu = jnp.array(nu)
    output_nu = hidden_nu

    hidden_variance = total_hidden_variance / max(num_layers - 1, 1)
    output_variance = jnp.array(1.0)

    hidden_kappa = jnp.array(1.0)
    output_kappa = hidden_kappa

    output_spherical_harmonic_fields = hidden_spherical_harmonic_fields = SphericalHarmonicFields(max_ell=max_ell)

    hidden_layers = []
    for _ in range(num_layers - 1):
        kernel = HodgeMaternKernel(
            kappa=hidden_kappa,
            nu=hidden_nu,
            variance=hidden_variance, 
            max_ell=max_ell,
        )
        prior = Prior(kernel=kernel)
        posterior = DummyPosterior(prior=prior)
        layer = SphericalHarmonicFieldsPosterior(posterior=posterior, spherical_harmonic_fields=hidden_spherical_harmonic_fields)
        hidden_layers.append(layer)

    kernel = HodgeMaternKernel(
        kappa=output_kappa,
        nu=output_nu,
        variance=output_variance, 
        max_ell=max_ell,
    )
    prior = Prior(kernel=kernel)
    likelihood = HodgeDeepGaussianLikelihood()
    posterior = Posterior(prior=prior, likelihood=likelihood)
    output_layer = SphericalHarmonicFieldsPosterior(posterior=posterior, spherical_harmonic_fields=hidden_spherical_harmonic_fields)
    return VectorOutputHodgeResidualDeepGP(hidden_layers=hidden_layers, output_layer=output_layer, num_samples=num_samples)


@jax.jit 
def hodge_expected_log_likelihood(
    y: Float[Array, "N 2"], 
    m: Float[Array, "N 2"], 
    f_var: Float[Array, "N 2 2"], 
    eps_var: Float[Array, ""],
) -> Float[Array, ""]:
    d = 2
    log2pi = jnp.log(2 * jnp.pi)
    squared_error = jnp.sum(jnp.square(y - m), axis=1) # sum([N 2] - [N 2], axis=1) -> [N]

    pointwise_val = (
        d * (log2pi + jnp.log(eps_var))
        + (squared_error + jnp.trace(f_var, axis1=1, axis2=2)) / eps_var # ([N] + [N]) / [1] -> [N]
    )
    return -0.5 * jnp.sum(pointwise_val, axis=0)


@Partial(jax.jit, static_argnames=('n',))
def hodge_deep_negative_elbo(p, x: Float, y: Float, *, key: Key, n: int) -> Float:
    eps_var = p.posterior.likelihood.noise_variance
    sample_keys = jax.random.split(key, p.num_samples)

    def sample_expected_log_likelihood(key: Key) -> Float:
        m, f_var = p.sample_moments(x, key=key)
        return hodge_expected_log_likelihood(y, m, f_var, eps_var)
    
    deep_expected_log_likelihood = jnp.mean(jax.vmap(sample_expected_log_likelihood)(sample_keys), axis=0)
    batch_ratio_correction = n / x.shape[0]

    return -(deep_expected_log_likelihood * batch_ratio_correction - p.prior_kl())


@jax.tree_util.Partial(jax.jit, static_argnames=("with_replacement",))
def closest_point_mask(targets: Array, x: Array, with_replacement: bool) -> Array:
    """
    Args: 
        targets (Array): targets in cartesian coordinates.
        x (Array): points in cartesian coordinates for which to produce the mask.
    """

    # Can do euclidean squared distance instead of spherical, since minimisation is invariant to monotonic transformations
    distances = jnp.sum((targets[:, None] - x[None, :]) ** 2, -1)

    def closest_point_mask_with_replacement():
        return jnp.argmin(distances, axis=1)
    
    def closest_point_mask_without_replacement():
        num_targets = targets.shape[0]
        closest_indices = jnp.zeros(num_targets, dtype=jnp.int64)
        available_mask = jnp.ones(x.shape[0], dtype=bool)

        for i in range(num_targets):
            masked_distances = jnp.where(available_mask, distances[i], jnp.inf)
            closest_idx = jnp.argmin(masked_distances)
            closest_indices = closest_indices.at[i].set(closest_idx)
            available_mask = available_mask.at[closest_idx].set(False)

        return closest_indices
    
    mask_indices = jax.lax.cond(
        with_replacement, 
        closest_point_mask_with_replacement, 
        closest_point_mask_without_replacement,
    )
    mask = jnp.zeros(x.shape[0], dtype=jnp.bool)
    return mask.at[mask_indices].set(True)


def angles_to_radians_colat(x: Array) -> Array:
    return jnp.pi * x / 180 + jnp.pi / 2

def angles_to_radians_lon(x: Array) -> Array:
    return jnp.pi * x / 180 

def angles_to_radians(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        colat=lambda df: angles_to_radians_colat(df.colat),
        lon=lambda df: angles_to_radians_lon(df.lon),
    )

def radians_to_angles_colat(x: Array) -> Array:
    return 180 * x / jnp.pi - 90 

def radians_to_angles_lon(x: Array) -> Array:
    return 180 * x / jnp.pi 

def radians_to_angles(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        colat=lambda df: radians_to_angles_colat(df.colat),
        lon=lambda df: radians_to_angles_lon(df.lon),
    )


era5_file_path = "../data/era5.nc"

era5_dataset = netCDF4.Dataset(era5_file_path,'r')
era5_lon = angles_to_radians_lon(era5_dataset.variables['longitude'][:].data.astype(np.float64))
era5_colat = angles_to_radians_colat(era5_dataset.variables['latitude'][:].data.astype(np.float64))
era5_lon_mesh, era5_colat_mesh = jnp.meshgrid(era5_lon, era5_colat)


def read_era5(time: int, level: int) -> pd.DataFrame:
    level = {
        0: 2, 
        7: 1, 
        15: 0, 
    }[level]
    u = era5_dataset.variables['u'][time, level].data.astype(np.float64)
    v = era5_dataset.variables['v'][time, level].data.astype(np.float64)
    df = pd.DataFrame({
        "lon": era5_lon_mesh.flatten(),
        "colat": era5_colat_mesh.flatten(),
        "u": u.flatten(),
        "v": v.flatten(),
    })
    return df


def match_to_uniform_grid_mask(x: Array, n: int, with_replacement: bool = True) -> Array:
    """
    Args: 
        x (Array): Points in cartesian coordinates for which to create the mask.
    """
    return closest_point_mask(
        targets=sphere_uniform_grid(n),
        x=x,
        with_replacement=with_replacement,
    )


def to_test_dataframe(df: pd.DataFrame, n: int, with_replacement: bool = True) -> pd.DataFrame:
    sph = df[['colat', 'lon']].values
    mask = match_to_uniform_grid_mask(
        x=sph_to_car(sph), n=n, with_replacement=with_replacement,
    ).tolist()
    return df[mask]



import pandas as pd 
import math 
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite, utc
from skyfield.toposlib import wgs84


def datetime_range(start, stop, step=timedelta(minutes=1)):
    current = start
    while current < stop:
        yield current
        current += step


def load_aeolus_and_timescale():
    ts = load.timescale()

    # Aeolus TLE data
    line1 = "1 43600U 18066A   21153.73585495  .00031128  00000-0  12124-3 0  9990"
    line2 = "2 43600  96.7150 160.8035 0006915  90.4181 269.7884 15.87015039160910"

    aeolus = EarthSatellite(line1, line2, "AEOLUS", ts)
    return aeolus, ts


def read_aeolus(start: datetime, stop: datetime, step=timedelta(minutes=1)) -> pd.DataFrame:
    if start.tzinfo is None:
        start = start.replace(tzinfo=utc)
    if stop.tzinfo is None:
        stop = stop.replace(tzinfo=utc)

    aeolus, ts = load_aeolus_and_timescale()
    time = list(datetime_range(start, stop, step))
    lat, lon = wgs84.latlon_of(aeolus.at(ts.from_datetimes(time)))

    # convert to colatitude [0, pi] and longitude [0, 2pi]
    colat, lon = lat.radians + math.pi / 2, lon.radians + math.pi

    return pd.DataFrame({
        "time": time,
        "colat": colat,
        "lon": lon,
    })

def to_train_dataframe(aeolus: pd.DataFrame, era5: pd.DataFrame, with_replacement: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = sph_to_car(aeolus[['colat', 'lon']].values)
    x = sph_to_car(era5[['colat', 'lon']].values)
    mask = closest_point_mask(
        targets=targets, 
        x=x, 
        with_replacement=with_replacement,
    ).tolist()
    return era5[mask], era5[~np.array(mask)]


def to_train_test_dataframes(aeolus: pd.DataFrame, era5: pd.DataFrame, test_size: int, with_replacement: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, rest_df = to_train_dataframe(aeolus=aeolus, era5=era5, with_replacement=with_replacement)
    test_df = to_test_dataframe(rest_df, n=test_size, with_replacement=with_replacement)
    return train_df, test_df


def train_test_sets(
    time: int, 
    level: int, 
    start: datetime, 
    stop: datetime, 
    step: timedelta, 
    test_size: int, 
    with_replacement: bool = True,
) -> tuple[Array, Array, Array, Array]:
    aeolus = read_aeolus(start=start, stop=stop, step=step)
    era5 = read_era5(time, level)
    
    # split data
    df_train, df_test = to_train_test_dataframes(
        aeolus=aeolus, era5=era5, test_size=test_size, with_replacement=with_replacement
    )

    # Inputs and target
    X_train, X_test = df_train[["colat", "lon"]].to_numpy(), df_test[["colat", "lon"]].to_numpy()
    y_train, y_test = df_train[["v", "u"]].to_numpy(), df_test[["v", "u"]].to_numpy()

    # Convert to jnp arrays (not sure if this is necessary)
    X_train, X_test = jnp.array(X_train), jnp.array(X_test)
    y_train, y_test = jnp.array(y_train), jnp.array(y_test)

    # Normalize (sort of) targets
    norm_constant = jnp.mean(jax.vmap(jnp.linalg.norm)(y_train))
    y_train /= norm_constant
    y_test /= norm_constant
    return X_train, X_test, y_train, y_test



from beartype.typing import (
    Callable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import jax
import optax as ox

from gpjax.base import Module
from gpjax.dataset import Dataset
from gpjax.objectives import AbstractObjective
from gpjax.scan import vscan
from gpjax.typing import ScalarFloat 

ModuleModel = TypeVar("ModuleModel", bound=Module)



def fit(  # noqa: PLR0913
    *,
    model: ModuleModel,
    objective: Union[AbstractObjective, Callable[[ModuleModel, Dataset], ScalarFloat]],
    x: Float, 
    y: Float,
    optim: ox.GradientTransformation,
    key: Key,
    num_iters: Optional[int] = 100,
    batch_size: Optional[int] = -1,
    verbose: Optional[bool] = True,
    unroll: Optional[int] = 1,
) -> Tuple[ModuleModel, Array]:
    r"""Train a Module model with respect to a supplied Objective function.
    Optimisers used here should originate from Optax.

    Example:
    ```python
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import optax as ox
        >>> import gpjax as gpx
        >>>
        >>> # (1) Create a dataset:
        >>> X = jnp.linspace(0.0, 10.0, 100)[:, None]
        >>> y = 2.0 * X + 1.0 + 10 * jr.normal(jr.key(0), X.shape)
        >>> D = gpx.Dataset(X, y)
        >>>
        >>> # (2) Define your model:
        >>> class LinearModel(gpx.base.Module):
                weight: float = gpx.base.param_field()
                bias: float = gpx.base.param_field()

                def __call__(self, x):
                    return self.weight * x + self.bias

        >>> model = LinearModel(weight=1.0, bias=1.0)
        >>>
        >>> # (3) Define your loss function:
        >>> class MeanSquareError(gpx.objectives.AbstractObjective):
                def evaluate(self, model: LinearModel, train_data: gpx.Dataset) -> float:
                    return jnp.mean((train_data.y - model(train_data.X)) ** 2)
        >>>
        >>> loss = MeanSqaureError()
        >>>
        >>> # (4) Train!
        >>> trained_model, history = gpx.fit(
                model=model, objective=loss, train_data=D, optim=ox.sgd(0.001), num_iters=1000
            )
    ```

    Args:
        model (Module): The model Module to be optimised.
        objective (Objective): The objective function that we are optimising with
            respect to.
        train_data (Dataset): The training data to be used for the optimisation.
        optim (GradientTransformation): The Optax optimiser that is to be used for
            learning a parameter set.
        num_iters (Optional[int]): The number of optimisation steps to run. Defaults
            to 100.
        batch_size (Optional[int]): The size of the mini-batch to use. Defaults to -1
            (i.e. full batch).
        key (Optional[KeyArray]): The random key to use for the optimisation batch
            selection. Defaults to jr.key(42).
        log_rate (Optional[int]): How frequently the objective function's value should
            be printed. Defaults to 10.
        verbose (Optional[bool]): Whether to print the training loading bar. Defaults
            to True.
        unroll (int): The number of unrolled steps to use for the optimisation.
            Defaults to 1.

    Returns
    -------
        Tuple[Module, Array]: A Tuple comprising the optimised model and training
            history respectively.
    """

    # Unconstrained space loss function with stop-gradient rule for non-trainable params.
    def loss(model: Module, x: Float, y: Float, *, key: Key) -> ScalarFloat:
        model = model.stop_gradient()
        return objective(model.constrain(), x, y, key=key)

    # Unconstrained space model.
    model = model.unconstrain()

    # Initialise optimiser state.
    state = optim.init(model)

    # Mini-batch random keys to scan over.
    iter_keys = jax.random.split(key, num_iters)

    # Optimisation step.
    def step(carry, key):
        model, opt_state = carry

        batch_x, batch_y = x, y

        loss_val, loss_gradient = jax.value_and_grad(loss)(model, batch_x, batch_y, key=key)
        updates, opt_state = optim.update(loss_gradient, opt_state, model)
        model = optax.apply_updates(model, updates)

        carry = model, opt_state
        return carry, loss_val

    # Optimisation scan.
    scan = vscan if verbose else jax.lax.scan

    # Optimisation loop.
    (model, _), history = scan(step, (model, state), (iter_keys), unroll=unroll)

    # Constrained space.
    model = model.constrain()

    return model, history


import argparse
import os 


num_iters = 1000
num_samples = 3
test_size = 5000
batch_size = -1 
save_dir = './results/wind'
num_hours = 24
step_minutes = 1
total_hidden_variance = 0.0001
num_test_samples = 10
max_ell_variational = 9

lr = 0.01


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--level", type=int, default=15, choices=[0, 7, 15])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=1)
    args = parser.parse_args()
    seed = args.seed
    time = args.time
    level = args.level
    num_layers = args.num_layers

    ### RANDOMNESS 
    key = jax.random.key(seed)
    data_key, train_key, test_key, plot_key = jax.random.split(key, 4)


    ### DATA     
    # aeolus track
    start = datetime(2019, 1, 1, 9)
    stop = start + timedelta(hours=num_hours)
    step = timedelta(minutes=step_minutes)
    train_size = (stop - start) // step 
    with_replacement = True

    # load 
    X_train, X_test, y_train, y_test = train_test_sets(time, level, start, stop, step, test_size, with_replacement)


    ### MODEL
    # settings 
    hidden_variance = total_hidden_variance / max(num_layers - 1, 1)
    obs_variance = 1.0
    experiment_name = f"{time=}-{level=}-{num_layers=}-{seed=}"
    print(f"Running experiment: {experiment_name}")
    # build 

    model = create_hodge_residual_deep_gp_with_spherical_harmonic_features(
        num_layers=num_layers,
        total_hidden_variance=total_hidden_variance,
        max_ell=max_ell_variational,
        x=X_train,
        num_samples=num_samples,
        key=train_key,
        nu=1.5,
        kernel_max_ell=None,
    )


    ### FIT
    # train 
    objective = Partial(hodge_deep_negative_elbo, n=train_size)

    optim = optax.adam(learning_rate=lr)
    model_opt, history = fit(
        model=model, 
        objective=objective, 
        x=X_train, 
        y=y_train, 
        optim=optim, 
        key=key, 
        num_iters=num_iters,
    )


    model_opt = model_opt.replace(num_samples=num_test_samples)

    def mse(y_true: Array, py: tfd.MixtureSameFamily) -> Float[Array, ""]:
        return jnp.mean(jnp.sum(jnp.square(y_true - py.mean()), axis=-1))


    def nlpd(y_true, py: tfd.MixtureSameFamily) -> Float[Array, ""]:
        return -jnp.mean(py.log_prob(y_true))


    def evaluate(key: Key, model, x_test, y_test):
        py = model.posterior.likelihood.diag(model.diag(x_test, key=key))
        return {
            'mse': mse(y_test, py).item(), 
            'nlpd': nlpd(y_test, py).item(),
        }

    test_metrics = evaluate(test_key, model_opt, X_test, y_test)

    metrics_str = ", ".join(f"{k}: {v:.3f}" for k, v in test_metrics.items())
    print(f"Metrics: {metrics_str}")

    ### SAVE RESULTS 

    # save
    experiment_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # metrics 
    metrics_path = os.path.join(experiment_dir, "metrics.csv")
    pd.DataFrame([test_metrics]).to_csv(metrics_path, index=False)
