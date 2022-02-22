#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J wavelet

cd /PATH/TO/ROOT/
source ~/.bashrc
conda activate wavelets
# Make comparable to the normal one
./run.sh experiments/v2_test_awr.py Acrobot-v1 0.0015 1.0 0 1 &

wait;
