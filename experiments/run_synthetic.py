from jax import config 
config.update("jax_enable_x64", True)


import jax 
import jax.numpy as jnp
import numpy as np
import pandas as pd 
import gpjax 
from typing import Optional, Callable
from jaxtyping import Array, Key
from gpjax.typing import Float, ScalarFloat
from jax.tree_util import Partial
import tensorflow_probability.substrates.jax.distributions as tfd
from utils import sphere_uniform_grid, car_to_sph, sph_to_car, sphere_meshgrid
from models import create_model, deep_negative_elbo


# def mse(y_true: Float[Array, "N"], y_pred: Float[Array, "N"]) -> Array:
#     return jnp.mean(jnp.square(y_true - y_pred))


# def nlpd(y_true, y_pred, std_pred):
#     return -jnp.mean(
#         tfd.Normal(loc=y_pred, scale=std_pred).log_prob(y_true)
#     )


# Copyright 2023 The JaxGaussianProcesses Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from beartype.typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import jax
import jax.random as jr
import optax as ox

from gpjax.base import Module
from gpjax.dataset import Dataset
from gpjax.objectives import AbstractObjective
from gpjax.scan import vscan

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
    iter_keys = jr.split(key, num_iters)

    # Optimisation step.
    def step(carry, key):
        model, opt_state = carry

        if batch_size != -1:
            batch_x, batch_y = get_batch(x, y, batch_size, key)
        else:
            batch_x, batch_y = x, y

        loss_val, loss_gradient = jax.value_and_grad(loss)(model, batch_x, batch_y, key=key)
        updates, opt_state = optim.update(loss_gradient, opt_state, model)
        model = ox.apply_updates(model, updates)

        carry = model, opt_state
        return carry, loss_val

    # Optimisation scan.
    scan = vscan if verbose else jax.lax.scan

    # Optimisation loop.
    (model, _), history = scan(step, (model, state), (iter_keys), unroll=unroll)

    # Constrained space.
    model = model.constrain()

    return model, history


def get_batch(x: Float, y: Float, batch_size: int, key: Key) -> tuple[Float, Float]:
    """Batch the data into mini-batches. Sampling is done with replacement.

    Args:
        train_data (Dataset): The training dataset.
        batch_size (int): The batch size.
        key (KeyArray): The random key to use for the batch selection.

    Returns
    -------
        Dataset: The batched dataset.
    """
    n = x.shape[0]

    # Subsample mini-batch indices with replacement.
    indices = jax.random.choice(key, n, (batch_size,), replace=True)

    return x[indices], y[indices]


import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from scipy.special import sph_harm


def rotate(sph, roll: float):
    """
    Apply a roll rotation to the points in spherical coordinates (colatitude, longitude) in [0, pi] x [0, 2pi].
    
    Parameters:
    colatitude (float or jnp.ndarray): Colatitude values in the range [0, pi].
    longitude (float or jnp.ndarray): Longitude values in the range [0, 2pi].
    angle (float): Angle of rotation in radians.
    
    Returns:
    jnp.ndarray: Array of transformed colatitude and longitude values.
    """
    colatitude, longitude = sph[..., 0], sph[..., 1]

    # Convert to Cartesian coordinates
    x = jnp.sin(colatitude) * jnp.cos(longitude)
    y = jnp.sin(colatitude) * jnp.sin(longitude)
    z = jnp.cos(colatitude)
    
    # Apply roll rotation
    new_x = x
    new_y = jnp.cos(roll) * y - jnp.sin(roll) * z
    new_z = jnp.sin(roll) * y + jnp.cos(roll) * z
    
    # Convert back to spherical coordinates
    new_colatitude = jnp.arccos(new_z / jnp.sqrt(new_x**2 + new_y**2 + new_z**2))
    new_longitude = jnp.arctan2(new_y, new_x)
    
    return jnp.stack([new_colatitude, new_longitude], axis=-1)


def reversed_spherical_harmonic(sph, m: int, n: int):
    colat, lon = sph[..., 0], sph[..., 1]
    return jnp.asarray(sph_harm(m, n, np.asarray(colat), np.asarray(lon)).real)


def target_f(sph: Float) -> Float:
    return reversed_spherical_harmonic(sph, m=1, n=2) + reversed_spherical_harmonic(rotate(sph, roll=jnp.pi / 2), m=1, n=1)


def add_noise(f: Float[Array, " N"], noise_std: float = 0.01, *, key: Key) -> Float:
    return f + jax.random.normal(key=key, shape=f.shape) * noise_std


def mse(y: Float[Array, "N"], py: tfd.MixtureSameFamily) -> Float:
    return jnp.mean(jnp.square(y - py.mean()))


def nlpd(y: Float[Array, "N"], py: tfd.MixtureSameFamily) -> Float:
    return -jnp.mean(py.log_prob(y))


def evaluate_diag(model: Module, x: Float[Array, "N D"], y: Float[Array, " N"], *, key: Key) -> dict[str, float]:
    py = model.posterior.likelihood.diag(model.diag(x, key=key))
    # mean, stddev = py.mean(), py.stddev()
    metrics = {
        # "mse": mse(y, mean).item(),
        # "nlpd": nlpd(y, mean, stddev).item(),
        "mse": mse(y, py).item(),
        "nlpd": nlpd(y, py).item(),
        # "nlpd": -jnp.mean(py.log_prob(y)).item(),
    }
    return metrics 


from typing import TypeVar 
from jaxtyping import Float
from gpjax.typing import ScalarFloat


Kernel = TypeVar("Kernel", bound="gpjax.kernels.base.AbstractKernel")



EPS = 1e-12



num_test = 5000
num_plot = 100


total_hidden_variance = 0.0001
num_inducing = 50
train_num_samples = 3
test_num_samples = 10

lr = 0.01
num_iters = 1000


def train_and_eval(
    num_train: int, model_name: str, num_layers: int, seed: int,
    num_inducing: int = 50, kernel_max_ell: int | None = None, 
): 
    key = jax.random.key(seed)
    key, data_key = jax.random.split(key)

    # data 
    train_x = sphere_uniform_grid(num_train)
    test_x = sphere_uniform_grid(num_test)
    plot_sph = jnp.stack(sphere_meshgrid(num_plot, num_plot), axis=-1).reshape(-1, 2)
    plot_x = sph_to_car(plot_sph)

    target_function = lambda x: target_f(car_to_sph(x))
    train_key, test_key, plot_key = jax.random.split(data_key, 3)
    train_y = add_noise(target_function(train_x), key=train_key)
    test_y = add_noise(target_function(test_x), key=test_key)

    if "hodge" in model_name:
        train_x = car_to_sph(train_x)
        test_x = car_to_sph(test_x)
        plot_x = car_to_sph(plot_x)

    # model
    key, model_key = jax.random.split(key)
    model = create_model(
        num_layers=num_layers, total_hidden_variance=total_hidden_variance, 
        num_inducing=num_inducing, x=train_x, num_samples=train_num_samples, 
        name=model_name, key=model_key, kernel_max_ell=kernel_max_ell
    )

    # training
    objective = Partial(deep_negative_elbo, n=num_train)
    optim = ox.adam(learning_rate=lr)
    model_opt, history = fit(
        model=model, 
        objective=objective, 
        x=train_x, 
        y=train_y, 
        optim=optim, 
        key=key, 
        num_iters=num_iters,
    )


    # testing
    model_opt = model_opt.replace(num_samples=test_num_samples)
    metrics_diag = evaluate_diag(model_opt, test_x, test_y, key=key)


    # plotting 
    # true function, mean, variance, error, history
    return model_opt, history, metrics_diag


import os 


def save_results(experiment_settings: dict[str, Any], metrics: dict[str, float], *, dir_path: str):
    experiment_name = "-".join(f"{k}={v}" for k, v in experiment_settings.items())
    experiment_dir = os.path.join(dir_path, experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)

    # save metrics
    metrics_file_name = os.path.join(experiment_dir, "metrics.csv")
    pd.DataFrame([experiment_settings | metrics]).to_csv(metrics_file_name, index=False, header=True)
    return 


def main(num_train, model_name, num_layers, seed, num_inducing: int = 50, kernel_max_ell: int | None = None):
    experiment_settings = {
        "num_train": num_train,
        "model_name": model_name,
        "num_layers": num_layers,
        "num_inducing": num_inducing,
        "seed": seed,
        "kernel_max_ell": kernel_max_ell,
    }
    print(experiment_settings)

    model_opt, history, metrics = train_and_eval(
        num_train, model_name, num_layers, seed, num_inducing=num_inducing, kernel_max_ell=kernel_max_ell
    )
    print(metrics)
    save_results(
        experiment_settings=experiment_settings,
        metrics=metrics,
        dir_path="results/synthetic",
    )


if __name__ == "__main__":
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_inducing", type=int, default=49)
    parser.add_argument("--kernel_max_ell", type=int, default=None)
    args = parser.parse_args()

    main(
        num_train=args.num_train,
        model_name=args.model_name,
        num_layers=args.num_layers,
        seed=args.seed,
        num_inducing=args.num_inducing,
        kernel_max_ell=args.kernel_max_ell,
    )