#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=1  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8GB       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-12:00:00     # DD-HH:MM:SS
##SBATCH --partition=unkillable
#SBATCH -o ./scratch/slurm_out/slurm-%j-%a.out
#SBATCH -e ./scratch/slurm_out/slurm-%j-%a.err

#SBATCH --array=1-6 # should match lines in commands.txt

module load python/3.10.lua
module load cudatoolkit/12.3.2
source ./venv/bin/activate

export WANDB_DIR=$SCRATCH/wandb

# mkdir $SLURM_TMPDIR/data
# tar xf ~/projects/def-xxxx/data.tar -C $SLURM_TMPDIR/data

echo "Starting task $SLURM_ARRAY_TASK_ID"
command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $PWD/scripts/commands.txt)
eval_command=$(eval echo $command)
srun $eval_command

#$SLURM_TMPDIR/data
# cp -R $SLURM_TMPDIR/ ~/scratch/
