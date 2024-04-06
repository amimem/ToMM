#!/bin/bash
#SBATCH --gres=gpu:rtx8000:1       # Request GPU "generic resources"

#SBATCH --cpus-per-task=2  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=80GB       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-6:00:00     # DD-HH:MM:SS
#SBATCH -o ~/scratch/slurm-%j-%a.out
#SBATCH -e ~/scratch/slurm-%j-%a.err

#SBATCH --array=0-1

module load python/3.10.lua
module load cudatoolkit/12.3.2
source ./venv/bin/activate

export WANDB_DIR=$SCRATCH/wandb

# mkdir $SLURM_TMPDIR/data
# tar xf ~/projects/def-xxxx/data.tar -C $SLURM_TMPDIR/data

# data generation
# python data_gen.py --N 10 --corr 1.0 --output $SCRATCH/output
# python data_gen.py --N 100 --corr 1.0 --output $SCRATCH/output
# python data_gen.py --N 1000 --corr 1.0 --output $SCRATCH/output
# python data_gen.py --N 10000 --corr 1.0 --output $SCRATCH/output

# python data_gen.py --N 10 --corr 0.5 --output $SCRATCH/output
# python data_gen.py --N 100 --corr 0.5 --output $SCRATCH/output
# python data_gen.py --N 1000 --corr 0.5 --output $SCRATCH/output
# python data_gen.py --N 10000 --corr 0.5 --output $SCRATCH/output

# python data_gen.py --N 10 --corr 0.0 --output $SCRATCH/output
# python data_gen.py --N 100 --corr 0.0 --output $SCRATCH/output
# python data_gen.py --N 1000 --corr 0.0 --output $SCRATCH/output
# python data_gen.py --N 10000 --corr 0.0 --output $SCRATCH/output

# scratch/output/data_75816d80b9 # 10
# scratch/output/data_330fb6f722 # 100
# scratch/output/data_45dd5039c4 # 1000
# scratch/output/data_2a6573283c # 10000
# python command_run.py

echo "Starting task $SLURM_ARRAY_TASK_ID"
command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $HOME/abstraction/scripts/commands.txt)
eval_command=$(eval echo $command)
srun $eval_command

#$SLURM_TMPDIR/data
# cp -R $SLURM_TMPDIR/ ~/scratch/
