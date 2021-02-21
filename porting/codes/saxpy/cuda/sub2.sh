#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=project_2002078
#SBATCH --partition=gputest
#SBATCH --time=00:05:00
#SBATCH -e error_%j
#SBATCH -o out_%j
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1


time srun ./saxpy
