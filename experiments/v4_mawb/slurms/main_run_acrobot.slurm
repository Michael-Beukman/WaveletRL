#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J pro_mawb_acro

cd /PATH/TO/ROOT/
source ~/.bashrc
conda activate wavelets
./run.sh experiments/v4_mawb/v4_run_main.py Acrobot-v1 0.00015 1.0 0.1 0 5 1 &

wait;
