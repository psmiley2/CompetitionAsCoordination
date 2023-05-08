#!/usr/bin/bash
#SBATCH --time=99:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBACTH --job-name="slurm test"

python3 /cluster/home/psmile01/e_200_b/driver1.py
