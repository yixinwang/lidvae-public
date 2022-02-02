#!/bin/sh

#Slurm directives
#
#SBATCH -A stats                 # The account name for the job.
#SBATCH -c 4                     # The number of cpu cores to use.
#SBATCH -t 11:55:00                  # The time the job will take to run.
#SBATCH --mem-per-cpu 6gb        # The memory the job will use per cpu core.

module load R

#Command to execute R code

echo "R CMD BATCH --no-save --vanilla ${FILENAME} ${OUTNAME}"

R CMD BATCH --no-save --vanilla ${FILENAME} ${OUTNAME}

# End of script