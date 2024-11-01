from jax import config 
config.update("jax_enable_x64", True)


from gpjax.typing import Float
from jaxtyping import Array, Key, ArrayLike
from numpy.typing import NDArray

import os 
import jax 
import jax.numpy as jnp
import numpy as np
import pandas as pd 
import optax 
from jax.tree_util import Partial
import tensorflow_probability.substrates.jax.distributions as tfd


# TODO credit gpjax, since we modify their fit function 
from beartype.typing import (
    Any,
    Optional,
    Tuple,
)
from gpjax.base import Module 
from gpjax.scan import vscan
from models import create_model, deep_negative_elbo
from pathlib import Path 
from scipy.io import arff



def fit(  # noqa: PLR0913
    *,
    model: Module, # TODO change to deep gp 
    objective,
    x: Float, 
    y: Float,
    optim: optax.GradientTransformation,
    key: Key,
    num_iters: Optional[int] = 100,
    batch_size: Optional[int] = None,
    verbose: Optional[bool] = True,
    unroll: Optional[int] = 1,
) -> Tuple[Module, Float]:

    # Unconstrained space loss function with stop-gradient rule for non-trainable params.
    def loss(model: Module, x: Float, y: Float, *, key: Key) -> Float:
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

        if batch_size is not None:
            batch_x, batch_y = get_batch(x, y, batch_size, key)
        else:
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


def mse_projected(y: Float[Array, "N"], py: tfd.MixtureSameFamily, x_norm: Float[Array, "N"]) -> Float:
    return jnp.mean(jnp.square(y - py.mean()) * jnp.square(x_norm))


def nlpd_projected(y: Float[Array, "N"], py: tfd.MixtureSameFamily, x_norm: Float[Array, "N"]) -> Float:
    return -jnp.mean(py.log_prob(y) - jnp.log(x_norm))


def evaluate_diag_projected(
    model: Module, x: Float[Array, "N D"], y: Float[Array, " N"], *, key: Key, x_norm: Float[Array, "N"]
) -> dict[str, float]:
    py = model.posterior.likelihood.diag(model.diag(x, key=key))
    metrics = {
        "mse": mse_projected(y, py, x_norm).item(),
        "nlpd": nlpd_projected(y, py, x_norm).item(),
    }
    return metrics 


data_dir = Path("../data/")


def read_yacht_data() -> tuple[NDArray, NDArray]:
    yacht_df = pd.read_csv(data_dir / f"yacht.data", header=None, sep='\s+').astype(np.float64)
    X, y = yacht_df.iloc[:, :-1].values, yacht_df.iloc[:, -1:].values
    return X, y 


def read_energy_data() -> tuple[NDArray, NDArray]:
    """
    We take the Heating Load as the target variable, which is second to last in the DataFrame.
    """
    energy_df = pd.read_excel(data_dir / f"energy.xlsx").astype(np.float64)
    X, y = energy_df.iloc[:, :-2].values, energy_df.iloc[:, -2:-1].values
    return X, y


def read_concrete_data() -> tuple[NDArray, NDArray]:
    concrete_df = pd.read_excel(data_dir / f"concrete.xls").astype(np.float64)
    X, y = concrete_df.iloc[:, :-1].values, concrete_df.iloc[:, -1:].values
    return X, y


def read_kin8mn_data() -> tuple[NDArray, NDArray]:
    data, _ = arff.loadarff(data_dir / f"kin8nm.arff")
    data = np.array(data.tolist(), dtype=np.float64)
    X, y = data[:, :-1], data[:, -1:]
    return X, y


def read_power_data() -> tuple[NDArray, NDArray]:
    power_df = pd.read_excel(data_dir / f"power.xlsx").astype(np.float64)
    X, y = power_df.iloc[:, :-1].values, power_df.iloc[:, -1:].values
    return X, y


def read_data(dataset: str) -> tuple[NDArray, NDArray]:
    if dataset == "yacht":
        return read_yacht_data()
    elif dataset == "energy":
        return read_energy_data()
    elif dataset == "concrete":
        return read_concrete_data()
    elif dataset == "kin8mn":
        return read_kin8mn_data()
    elif dataset == "power":
        return read_power_data()
    else:
        raise ValueError(f"Dataset {dataset} not found.")


def to_jax(*args: ArrayLike) -> tuple[Array, ...]:
    return tuple(jnp.asarray(arg, dtype=jnp.float64) for arg in args)


def train_test_split(X: Array, y: Array, key: Key) -> tuple[Array, Array, Array, Array]:
    """
    Split the data into training and testing sets with a 90-10 split.
    """
    n = X.shape[0]
    split = int(0.9 * n)

    perm = jax.random.permutation(key, n)
    X, y = X[perm], y[perm]
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    return X_train, X_test, y_train, y_test


def standardize(X_train: Array, X_test: Array, y_train: Array, y_test: Array) -> tuple[Array, Array, Array, Array]:
    """
    Standardize the training and testing data. Does not return the mean and standard deviation, 
    since the NLPD and MSE is reported on the standardized data.
    """
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    y_train_mean = y_train.mean(axis=0)
    y_train_std = y_train.std(axis=0)

    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    y_train = (y_train - y_train_mean) / y_train_std
    y_test = (y_test - y_train_mean) / y_train_std

    return X_train, X_test, y_train, y_test
    

def project_data_to_sphere(X: Array, y: Array, bias: float = 1.0):
    X_projected = jnp.empty((X.shape[0], X.shape[1] + 1), dtype=X.dtype)
    X_projected = X_projected.at[:, :-1].set(X)
    X_projected = X_projected.at[:, -1].set(bias)
    X_projected_norm = jnp.linalg.norm(X_projected, axis=1, keepdims=True)
    return X_projected / X_projected_norm, y / X_projected_norm, X_projected_norm


def train_and_eval(
    dataset_name: str, model_name: str, num_layers: int, seed: int, num_iters: int, kernel_max_ell: int | None = None, batch_size: int = None,
): 
    key = jax.random.key(seed)
    key, data_key = jax.random.split(key)

    # data 
    x, y = read_data(dataset_name)
    x, y = to_jax(x, y)
    train_x, test_x, train_y, test_y = train_test_split(x, y, key=data_key)
    train_x, test_x, train_y, test_y = standardize(train_x, test_x, train_y, test_y)

    if model_name.startswith("residual"):
        train_x, train_y, _ = project_data_to_sphere(train_x, train_y)
        test_x, test_y, test_x_norm = project_data_to_sphere(test_x, test_y)
        test_x_norm = test_x_norm.squeeze(axis=-1)
    else:
        test_x_norm = jnp.ones((test_x.shape[0]))
    
    train_y, test_y = train_y.squeeze(axis=-1), test_y.squeeze(axis=-1)

    # model
    key, model_key = jax.random.split(key)

    num_inducing = dataset_name_to_num_inducing[dataset_name]
    model = create_model(
        num_layers=num_layers, total_hidden_variance=total_hidden_variance, 
        num_inducing=num_inducing, x=train_x, num_samples=train_num_samples, 
        name=model_name, key=model_key, kernel_max_ell=kernel_max_ell,
    )

    # training
    optim = optax.adam(learning_rate=lr)
    if batch_size is not None and batch_size >= train_x.shape[0]:
        batch_size = None
    n = train_x.shape[0]
    objective = Partial(deep_negative_elbo, n=n)

    model_opt, history = fit(
        model=model, 
        objective=objective, 
        x=train_x, 
        y=train_y, 
        optim=optim, 
        key=key, 
        num_iters=num_iters,
        batch_size=batch_size,
    )


    # testing
    model_opt = model_opt.replace(num_samples=test_num_samples)
    metrics_diag = evaluate_diag_projected(model_opt, test_x, test_y, key=key, x_norm=test_x_norm)
    return model_opt, history, metrics_diag


def save_results(experiment_settings: dict[str, Any], metrics: dict[str, float], *, dir_path: str):
    experiment_name = "-".join(f"{k}={v}" for k, v in experiment_settings.items())
    experiment_dir = os.path.join(dir_path, experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)

    # save metrics
    metrics_file_name = os.path.join(experiment_dir, "metrics.csv")
    pd.DataFrame([experiment_settings | metrics]).to_csv(metrics_file_name, index=False, header=True)
    return 


def main(dataset_name: str, model_name: str, num_layers: int, seed: int, num_iters: int, batch_size: int = None):
    experiment_settings = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "num_layers": num_layers,
        "seed": seed,
        "kernel_max_ell": None,
        "num_iters": num_iters,
        "batch_size": batch_size,
    }
    print(experiment_settings)

    model_opt, _, metrics = train_and_eval(
        dataset_name=dataset_name, model_name=model_name, num_layers=num_layers, seed=seed,
        kernel_max_ell=None, num_iters=num_iters, batch_size=batch_size,
    )
    print(metrics)
    save_results(
        experiment_settings=experiment_settings,
        metrics=metrics,
        dir_path="results/uci/",
    )
    return model_opt


dataset_name_to_num_inducing = {
    "yacht": 294,
    "concrete": 210, 
    "energy": 210,
    "kin8mn": 210,
    "power": 336,
}
total_hidden_variance = 0.0001

lr = 0.01
train_num_samples = 3

num_test = 5000
test_num_samples = 10


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, required=True)
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--num_layers", type=int, required=True)
    argparser.add_argument("--seed", type=int, required=True)
    argparser.add_argument("--num_iters", type=int, default=5000)
    argparser.add_argument("--batch_size", type=lambda x: None if x.lower() == 'none' else int(x), default=None)

    args = argparser.parse_args()
    main(
        dataset_name=args.dataset, 
        model_name=args.model, 
        num_layers=args.num_layers, 
        seed=args.seed, 
        num_iters=args.num_iters,
        batch_size=args.batch_size,
    )