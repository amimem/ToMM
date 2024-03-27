#!/bin/bash

# Define arguments & values
model_names=("stomp" "single" "multi")
P=(1e5 1e6 1e7 1e8 1e9)
N=("sirenity" "of" "abstracted" "minds")
# N=(1e1 1e2 1e3 1e4)


# Generate Python commands for all combinations 
# of argument values & write to txt file
for model_name in "${model_names[@]}"; do
    for P in "${P[@]}"; do
        for N in "${N[@]}"; do
            echo "python train.py --model_name $model_name --P $P --N $N" >> commands.txt
        done
    done
done
