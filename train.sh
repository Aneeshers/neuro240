#!/bin/bash
#SBATCH -c 2                # Number of cores (-c)
#SBATCH -t 2-04:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner   # Partition to submit to
#SBATCH --account=kempner_gershman_lab
#SBATCH --mem=130000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH -o /n/home04/amuppidi/neuro240/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home04/amuppidi/neuro240/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# load modules
module load python/3.10.9-fasrc01
conda activate torch
cd /n/home04/amuppidi/neuro240

# run code with passed arguments
~/.conda/envs/torch/bin/python train.py
