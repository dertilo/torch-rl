#!/bin/bash

#SBATCH -o ./torch-rl-snake.log   # Output-File
#SBATCH -D .                  # Working Directory
#SBATCH -J RL-Snake 		# Job Name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40

##Max Walltime vorgeben:
#SBATCH --time=01:00:00 # Erwartete Laufzeit

#Auf Standard-Knoten rechnen:
#SBATCH --partition=standard

#Job-Status per Mail:
#SBATCH --mail-type=NONE
#SBATCH --mail-user=tilo.himmelsbach@tu-berlin.de

source activate torch-rl
cd torch-rl
python torch_rl/scripts/run_dqn.py