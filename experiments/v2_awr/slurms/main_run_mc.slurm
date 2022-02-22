#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J wavelet

cd /PATH/TO/ROOT/
source ~/.bashrc
conda activate wavelets
./run.sh experiments/v2_test_awr.py MountainCar-v0 0.015 0.5 2 1 &

wait;
