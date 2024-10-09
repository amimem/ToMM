import itertools

# Define different values for each argument
paras = {
    "N": [10],
    "corr": [0.8],
    "P": [int(5e5)],  # Assuming only one value as per the excerpt
    "seq_len": [16],  # Assuming only one value as per the excerpt
    "training_sample_budget": [int(1e4)],  # Assuming only one value as per the excerpt
    "use_pos_enc": [0, 1], 
    "inter": ['None','attn'],
    "S": [8],
    "A": [2],
    "wstate": [0., 1. ,10.],
    "batch_size": [8],  # Assuming only one value as per the excerpt
    "num_epochs": [100],  # Assuming only one value as per the excerpt
    "learning_rate": [5e-4],  # Assuming only one value as per the excerpt
    "data_seed": [0],  # Assuming only one value as per the excerpt
    "seed": [0],  # Assuming only one value as per the excerpt
    "evaluation_sample_size": [int(1e4)] , # Assuming only one value as per the excerpt
    "use_wandb": [1]
}
# Generate all combinations of argument values/
all_combinations = itertools.product(*paras.values())
# Open the file to write the commands
with open('scripts/commands.txt', 'w') as file:
    for combination in all_combinations:
        command = "python mlp_match.py " + ' '.join([f"--{key} {value}" for key,value in zip(paras.keys(),combination)])
        file.write(command + '\n')

print("Commands written to commands.txt")
