#!/bin/bash
#SBATCH --job-name=VIT
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12

nvidia-smi

python /home/abdelrahman.elsayed/VIT_from_scratch/main.py
