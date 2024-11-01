#!/bin/bash

python run_uci.py --dataset yacht --model euclidean+inducing_points --num_layers 1 --seed 0 --batch_size None --num_iters 5
python run_uci.py --dataset yacht --model euclidean+inducing_points --num_layers 2 --seed 0 --batch_size None --num_iters 5
python run_uci.py --dataset yacht --model residual+spherical_harmonic_features --num_layers 1 --seed 0 --batch_size None --num_iters 5
python run_uci.py --dataset yacht --model residual+spherical_harmonic_features --num_layers 2 --seed 0 --batch_size None --num_iters 5