#!/bin/bash
#SBATCH -J lsgm_pinn
#SBATCH -p amd_a100nv_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o lsgm_pinn_train.out
#SBATCH -e lsgm_pinn_train.err
#SBATCH --time 40:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch
#SBATCH --ntasks-per-node=1

python3 train.py