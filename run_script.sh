#!/bin/bash

# Change to the directory containing the Python script
# cd /mnt/c/Users/maxpu/Dropbox/scripts/Projects/MARL/marl/gumbel_partition

# data paras
N=4
K=5
corr=1.0
M=2

sps=16
T=512 # =sps*2^K

# factors=(1 2 4 8 16)
# factor=8
# for factor in "${factors[@]}"; do
FILEdata=_4agentdebug_modelname_bitpop_corr_"$corr"_ensemble_sum_M_"$M"_simulationdata_actsel_greedy_numepi_1_K_"$K"_N_"$N"_T_"$T"_sps_"$sps"
# # if [ -f "output/$FILEdata.npy" ]; then
# #     echo "$FILEdata.npy exists."
# # else 
python -u -W ignore main_data_generation.py --sps "$sps" --N "$N" --T "$T" --corr "$corr" --M "$M"
# # fi
# done

# training paras
# lr=0.0005
epochs=200
# for lr in 0.0001 0.0005 0.001 0.005 0.01; do # "STOMPnet_M_2_L_4_nfeatures_2" 
lr=0.00005
for model in "STOMPnet_M_2_L_100_nfeatures_2" "singletaskbaseline" "multitaskbaseline"; do
	for seed in 0 1 2; do 
	FILEtrain=_"$model"_cap_240_trainseed_0_epochs_"$epochs"_batchsz_16_lr_"$lr"
	# if [ -f "output/$FILEdata$FILEtrain.npy" ]; then
	    # echo "$FILEdata$FILEtrain.npy exists."
	# else 
	python -u -W ignore main_training.py --model_name "$model" --data_filename "$FILEdata" --learning_rate "$lr" --epochs "$epochs" --seed "$seed"
	# fi
done
done

# for factor in "${factors[@]}"; do
# 	python -u -W ignore main_training.py --model_name "multitaskbaseline" --data_filename "_4agentdebug_modelname_bitpop_corr_1.0_ensemble_sum_M_2_simulationdata_actsel_greedy_numepi_1_K_10_N_4_T_10000_g_"$factor".0"
# done

# for model in "singletaskbaseline" "multitaskbaseline"; do
# 	python -u -W ignore main_training.py --model_name "$model" --data_filename "_4agentdebug_modelname_bitpop_corr_0.8_ensemble_sum_M_2_simulationdata_actsel_greedy_numepi_1_K_10_N_4_T_10000_g_8.0"
# done