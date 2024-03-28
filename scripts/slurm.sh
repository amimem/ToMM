#!/bin/bash
#SBATCH --gres=gpu:rtx8000:1       # Request GPU "generic resources"

#SBATCH --cpus-per-task=2  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=80GB       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-6:00:00     # DD-HH:MM:SS
#SBATCH -o /home/mila/m/memariaa/scratch/slurm-%j-%a.out
#SBATCH -e /home/mila/m/memariaa/scratch/slurm-%j-%a.err
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=memariaa@mila.quebec

#SBATCH --array=0-18


module load python/3.10.lua
module load cudatoolkit/12.3.2
source ./venv/bin/activate

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

# scratch/output/data_e72dd17cbc # 10
# scratch/output/data_8cd3f1c6e8 # 100
# scratch/output/data_025501a652 # 1000
# scratch/output/data_66d5636af3 # 10000

python train.py --outdir $SCRATCH/output --data_dir data_e72dd17cbc --L 128 --P 1e6 --n_features 2 --num_codebook 64 --model_name "stomp"
python train.py --outdir $SCRATCH/output --data_dir data_8cd3f1c6e8 --L 256 --P 1e7 --n_features 32 --num_codebook 64 --model_name "stomp"
python train.py --outdir $SCRATCH/output --data_dir data_025501a652 --L 512 --P 1e8 --n_features 256 --num_codebook 256 --model_name "stomp"
python train.py --outdir $SCRATCH/output --data_dir data_66d5636af3 --L 1024 --P 1e9 --n_features 1024 --num_codebook 256 --model_name "stomp"

python train.py --outdir $SCRATCH/output --data_dir data_e72dd17cbc --L 128 --P 1e6 --n_features 2 --num_codebook 64 --model_name "single"
python train.py --outdir $SCRATCH/output --data_dir data_8cd3f1c6e8 --L 256 --P 1e7 --n_features 32 --num_codebook 64 --model_name "single"
python train.py --outdir $SCRATCH/output --data_dir data_025501a652 --L 512 --P 1e8 --n_features 256 --num_codebook 256 --model_name "single"
python train.py --outdir $SCRATCH/output --data_dir data_66d5636af3 --L 1024 --P 1e9 --n_features 1024 --num_codebook 256 --model_name "single"

python train.py --outdir $SCRATCH/output --data_dir data_e72dd17cbc --L 128 --P 1e6 --n_features 2 --num_codebook 64 --model_name "multi"
python train.py --outdir $SCRATCH/output --data_dir data_8cd3f1c6e8 --L 256 --P 1e7 --n_features 32 --num_codebook 64 --model_name "multi"
python train.py --outdir $SCRATCH/output --data_dir data_025501a652 --L 512 --P 1e8 --n_features 256 --num_codebook 256 --model_name "multi"
python train.py --outdir $SCRATCH/output --data_dir data_66d5636af3 --L 1024 --P 1e9 --n_features 1024 --num_codebook 256 --model_name "multi"

# python command_run.py

# echo "Starting task $SLURM_ARRAY_TASK_ID"
# command=$(sed –n "${SLURM_ARRAY_TASK_ID}p" /home/mila/m/memariaa/marl/sss)
# srun $command

#$SLURM_TMPDIR/data
# cp -R $SLURM_TMPDIR/ ~/scratch/
