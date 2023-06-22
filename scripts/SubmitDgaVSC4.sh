#!/bin/bash
#SBATCH -N 2
#SBATCH -J LambdaDga
#SBATCH --ntasks-per-node=48
#SBATCH --partition=skylake_0096
#SBATCH --qos=skylake_0096
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL                              # first have to state the type of event to occur
#SBATCH --mail-user=<p.worm@a1.net>   # and then your email address

. $HOME/Programs/module_files/w2dynamics_modules

mpiexec -np $SLURM_NTASKS dga_main > lambda_dga-$SLURM_JOB_ID.log 2>&1



