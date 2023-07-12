#!/bin/bash
#SBATCH -J vmap_pinn
#SBATCH -p amd_a100nv_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o vmap_pinn_train.out
#SBATCH -e vmap_pinn_train.err
#SBATCH --time 40:00:00
#SBATCH --gres=gpu:6
#SBATCH --comment pytorch
#SBATCH --ntasks-per-node=6

python3 train.py