#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J acro_ib_proper

cd /PATH/TO/ROOT/
source ~/.bashrc
conda activate wavelets
./run.sh experiments/v3_ibfdd/v3_grid_search.py Acrobot-v1 0.003 1.0 2 25 1 &

wait;
