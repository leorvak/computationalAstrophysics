#!/bin/bash

# the name of the job as it will appear in the queue
#SBATCH --job-name=mcmc_mpi
# name of the output file where all "screen" io will be written
#SBATCH --output=mcmc.out
# ask to allocate a given number of nodes (each node has 2 CPUs=18 cores)
#SBATCH --nodes=1
# as for a certain number of parallel tasks
# each parallel task will run on a separate core
#SBATCH --ntasks=4
# account to charge the time to
#SBATCH --account=astr33000
# indicate partition (ask for nodes with Sandybridge CPUs
#SBATCH --partition=sandyb

module unload python
module load mpi4py/1.3+intelmpi-4.0
mpirun python metropolis_mcmc_parallel.py

