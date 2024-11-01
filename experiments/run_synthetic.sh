#!/bin/bash

python run_synthetic.py --num_train 100 --model_name euclidean_with_geometric_input+inducing_points --num_layers 2 --seed 2 --num_inducing 49
python run_synthetic.py --num_train 100 --model_name residual+inducing_points --num_layers 2 --seed 2 --num_inducing 49
python run_synthetic.py --num_train 100 --model_name residual+spherical_harmonic_features --num_layers 2 --seed 2 --num_inducing 49
python run_synthetic.py --num_train 100 --model_name residual+hodge+spherical_harmonic_features --num_layers 2 --seed 2 --num_inducing 72