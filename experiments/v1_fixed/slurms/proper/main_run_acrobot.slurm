#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J wavelet

cd /PATH/TO/ROOT/
source ~/.bashrc
conda activate wavelets
./run.sh experiments/v1_fixed/v1_fixed.py Acrobot-v1 1.0 0.9 0.0 0.15 0.001 0.015 0.015 0.015 0.00015 0.00015 0.001 1
wait;
