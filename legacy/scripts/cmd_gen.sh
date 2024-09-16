#!/usr/bin/bash

# Define arguments & values


# model-specific arguments
model_names=(
    # "stomp" 
      "single" 
    # "multi" 
    # "decoderonly"
    )
model_learning_rates=(
    9e-5
    5e-6
    1e-4
    1e-4
    )
epochs=(
    200
    200
    200
    200
    )


# size-specific arguments
P=(
    1e6 
    1e7 
    1e8 
    1e9
    )
P_Ls=(
    1e10
    4e10
    16e10
    64e10
    256e10
    )


# system size specific arguments
N=(
    1e1 
    1e2 
    1e3 
    1e4
    )
N_Ks=(
    4
    5
    6
    7
    )


# run this on these commands cluster
# for nit in "${!N[@]}"; do
#     echo "python data_gen.py --ensemble sum --N ${N[$nit]} --corr 0.8 --K ${N_Ks[$nit]}" >> data_commands.txt

# to generate these datafiles:
Ndatadirs=(
    "scratch/output/data_a318de6e3b" # 10
    "scratch/output/data_e4ab8599eb" # 100
    "scratch/output/data_9f8c395810" # 1000
    "scratch/output/data_9d87d466ca" # 10000
    )
Ndatadirs=(
    "data_deb6fe1a1c"
    "data_9bb9ab5a87"
    "data_4d16c40ccd"
    "data_92f5d0f65c"
    )


# Generate Python commands for all combinations 
# of argument values & write to txt file
for mit in "${!model_names[@]}"; do
    for pit in "${!P[@]}"; do
        for nit in "${!Ndatadirs[@]}"; do # datadir in "${Ndatadirs[@]}"; do
            if [ $(($nit+$pit)) -lt 4 ]
            then
                learning_rate=1e-4 # $(echo "10^(-${nit}-${pit}-3)" | bc -l )
                echo "python train.py --model_name ${model_names[$mit]} --P ${P[$pit]} --L 100 --data_dir ${Ndatadirs[$nit]} --learning_rate ${learning_rate} --epochs ${epochs[$mit]}" >> commands.txt
            fi
        done
    done
done

# now run slurm.sh to submit the commands listed in commands.txt as job array
