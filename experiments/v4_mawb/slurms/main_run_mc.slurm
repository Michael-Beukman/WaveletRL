#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J pro_mawb_mc

cd /PATH/TO/ROOT/
source ~/.bashrc
conda activate wavelets
./run.sh experiments/v4_mawb/v4_run_main.py MountainCar-v0 0.01 0.5 0.1 1 5 1 &

wait;
