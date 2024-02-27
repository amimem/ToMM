#!/bin/bash

# Change to the directory containing the Python script
# cd /mnt/c/Users/maxpu/Dropbox/scripts/Projects/MARL/marl/gumbel_partition

# Run the Python script
factors=(1 2 3 4 5 6 7 8)
for factor in "${factors[@]}"
do
	echo python -u -W ignore main_data_generation.py --stablefac "$factor"
done

