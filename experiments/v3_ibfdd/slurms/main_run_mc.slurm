#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J mc_ib_proper

cd /PATH/TO/ROOT/
source ~/.bashrc
conda activate wavelets
./run.sh experiments/v3_ibfdd/v3_grid_search.py MountainCar-v0 0.0005 0.1 2 10 1 &

wait;
