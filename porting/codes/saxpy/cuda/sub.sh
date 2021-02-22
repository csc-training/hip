#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=4training
#SBATCH --reservation=gpu_training
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH -e error_%j
#SBATCH -o out_%j
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1


time srun ./saxpy
