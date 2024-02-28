#!/bin/bash

# Change to the directory containing the Python script
# cd /mnt/c/Users/maxpu/Dropbox/scripts/Projects/MARL/marl/gumbel_partition

# Run the Python script
factors=(1 2 4 8 16)
for factor in "${factors[@]}"; do
	python -u -W ignore main_data_generation.py --stablefac "$factor"
	for model in "STOMPnet_M_2_L_4_nfeatures_2" "singletaskbaseline" "multitaskbaseline"; do
		python -u -W ignore main_training.py --model_name "$model" --data_filename "_4agentdebug_modelname_bitpop_corr_1.0_ensemble_sum_M_2_simulationdata_actsel_greedy_numepi_1_K_10_N_4_T_10000_g_"$factor".0"
	done
done


