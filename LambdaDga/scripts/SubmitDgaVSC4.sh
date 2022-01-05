#!/bin/bash
#SBATCH -N 2
#SBATCH -J LambdaDga_Nk16
#SBATCH --ntasks-per-node=48
#SBATCH --partition=mem_0096
##SBATCH --qos=mem_0096
#SBATCH --qos=p71282_0096
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL                              # first have to state the type of event to occur
#SBATCH --mail-user=<p.worm@a1.net>   # and then your email address

mpiexec -np $SLURM_NTASKS python DgaMain.py > lambda_dga-$SLURM_JOB_ID.log 2>&1



