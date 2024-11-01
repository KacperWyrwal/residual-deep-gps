from jax import config 
config.update("jax_enable_x64", True)


from jaxtyping import Array, Key
from gpjax.typing import Float, ScalarFloat 


import jax 
import jax.numpy as jnp
import pandas as pd
import optax 
from gpjax.base import Module
from dataclasses import dataclass, field 
from pathlib import Path
from jax.tree_util import Partial 
import timeit 
from enum import Enum 
from abc import ABC 
from utils import sphere_uniform
from models import create_model, deep_negative_elbo


def time_function(func, globals_dict=None, number=1, repeat=100):
    assert isinstance(func, str)
    assert func.startswith("jax.block_until_ready(") or func.endswith(".block_until_ready()")

    if globals_dict is None:
        globals_dict = {}
    timer = timeit.Timer(func, globals=globals_dict)
    timer.timeit(number=1)
    return timer.repeat(repeat=repeat, number=number)


@dataclass 
class Dataset(ABC):
    name: str 
    dim: int
    num: int 
    num_inducing: int
    kernel_max_ell: int
    x: Float[Array, "N D"] = field(init=False)
    batch_size: int | None = field(default=None)

    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = int(self.num * 0.9)
        self.x_train = sphere_uniform(self.dim, self.batch_size, key=jax.random.key(0))
        self.y_train = jnp.zeros(self.batch_size)
    

@dataclass
class Yacht(Dataset):
    name: str = "yacht"
    dim: int = 6
    num: int = 308
    num_inducing: int = 294
    kernel_max_ell: int = 12


@dataclass
class Energy(Dataset):
    name: str = "energy"
    dim: int = 8
    num: int = 768
    num_inducing: int = 210
    kernel_max_ell: int = 10


@dataclass
class Concrete(Dataset):
    name: str = "concrete"
    dim: int = 8
    num: int = 1030
    num_inducing: int = 294
    kernel_max_ell: int = 12


@dataclass
class Kin8mn(Dataset):
    name: str = "kin8mn"
    dim: int = 8
    num: int = 8192
    num_inducing: int = 210
    kernel_max_ell: int = 10
    batch_size: int = 1000


@dataclass
class Power(Dataset):
    name: str = "power"
    dim: int = 4
    num: int = 9568
    num_inducing: int = 336
    kernel_max_ell: int = 20
    batch_size: int = 1000


class Datasets(Enum):
    yacht = Yacht()
    concrete = Concrete()
    energy = Energy()
    kin8mn = Kin8mn()
    power = Power()



total_hidden_variance = 0.0001
train_num_samples = 3
lr = 0.01



if __name__ == "__main__":
    import argparse 
    import os 

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yacht")
    parser.add_argument("--model", type=str, default="residual+spherical_harmonic_features")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dataset = args.dataset
    num_iters = args.num_iters
    num_layers = args.num_layers
    model_name = args.model
    seed = args.seed


    key = jax.random.key(seed)
    dataset = Datasets[dataset].value
    settings = {
        "dataset": dataset.name,
        "model": model_name,
        "num_layers": num_layers,
        "dataset_dim": dataset.dim,
        "batch_size": dataset.batch_size,
        "num_iters": num_iters,
        "num_inducing": dataset.num_inducing,
        "seed": seed,
    }

    print("Running with settings:")
    print(settings)

    x_train = dataset.x_train
    y_train = dataset.y_train
    model = create_model(
        num_layers, total_hidden_variance, dataset.num_inducing, x_train, 
        num_samples=train_num_samples, key=key, name=model_name
    )
    optim = optax.adam(lr)
    objective = Partial(deep_negative_elbo, n=dataset.batch_size)

    def loss(model: Module, x: Float, y: Float, *, key: Key) -> ScalarFloat:
        model = model.stop_gradient()
        return objective(model, x, y, key=key)
    
    grad_loss = jax.grad(loss)
    jit_grad_loss = jax.jit(grad_loss)

    # time 
    func = "jax.block_until_ready(jit_grad_loss(model, x_train, y_train, key=key))"
    times = time_function(func, globals_dict=locals(), number=1, repeat=num_iters)


    # save results 
    experiment_dir = Path("results/time_uci") / "-".join([f"{k}={v}" for k, v in settings.items()])
    os.makedirs(experiment_dir, exist_ok=True)
    result_path = experiment_dir / "results.csv"
    result = settings | {"time": times}
    pd.DataFrame([result]).explode('time').to_csv(result_path, index=False)
