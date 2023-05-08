#!/usr/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBACTH --job-name="slurm test"
#SBATCH --output=log.txt

python3 /cluster/home/psmile01/planaria_simulator/runExperiments.py
