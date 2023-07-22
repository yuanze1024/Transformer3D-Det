#!/bin/bash

#SBATCH -J vote_refine
#SBATCH -p gpu-quota
#SBATCH --gres=gpu:1
#SBATCH -o plain.out
#SBATCH -e plain.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

source activate pt112

mkdir -p log
python ../../../algorithm/distill.py
