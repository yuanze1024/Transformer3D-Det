#!/bin/bash

#SBATCH -J yuanze_refine_sunrgbd
#SBATCH -p gpu-quota
#SBATCH --gres=gpu:1
#SBATCH -o default_setting.out
#SBATCH -e default_setting.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

source activate pt112

mkdir -p log
python ../../../algorithm/distill.py
