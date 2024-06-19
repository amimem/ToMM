import itertools

# Define different values for each argument
N_values = [10, 100, 1000]
corr_values = [0, 0.5, 0.99]
P_values = [int(5e5)]  # Assuming only one value as per the excerpt
seq_len_values = [8]  # Assuming only one value as per the excerpt
training_sample_budget_values = [int(1e4)]  # Assuming only one value as per the excerpt
batch_size_values = [8]  # Assuming only one value as per the excerpt
num_epochs_values = [50]  # Assuming only one value as per the excerpt
learning_rate_values = [5e-4]  # Assuming only one value as per the excerpt
data_seed_values = [0]  # Assuming only one value as per the excerpt
seed_values = [0]  # Assuming only one value as per the excerpt
evaluation_sample_size_values = [int(1e4)]  # Assuming only one value as per the excerpt

# Generate all combinations of argument values
all_combinations = itertools.product(
    N_values, corr_values, P_values, seq_len_values, training_sample_budget_values,
    batch_size_values, num_epochs_values, learning_rate_values, data_seed_values,
    seed_values, evaluation_sample_size_values
)

# Open the file to write the commands
with open('scripts/commands.txt', 'w') as file:
    for combination in all_combinations:
        command = f"python mlp_match.py --N {combination[0]} --corr {combination[1]} --P {combination[2]} --seq_len {combination[3]} --training_sample_budget {combination[4]} --batch_size {combination[5]} --num_epochs {combination[6]} --learning_rate {combination[7]} --data_seed {combination[8]} --seed {combination[9]} --evaluation_sample_size {combination[10]}"
        file.write(command + '\n')

print("Commands written to commands.txt")