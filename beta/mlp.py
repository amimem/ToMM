import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import h5py
import yaml
import torch

# data_dir = "output/data_b4c7f51405" # 1 agents
data_dir = "output/data_78d5097045" # 10 agents
data_filename = f"{data_dir}/data.h5"
config_filename = f"{data_dir}/config.yaml"

# load the hdf data
with h5py.File(data_filename, 'r') as f:
    datasets = {}
    for group_name, group in f.items():
        datasets[group_name] = {key: np.array(
            value) for key, value in group.items()}

# load the config file
with open(config_filename, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config, flush=True)

data_seed = 0
data = datasets[f"dataset_{data_seed}"]

states = data["states"]
actions = data["actions"]

# shape of states: (num_states, dim_teacher_inp)
# shape of actions: (num_states, num_teachers)
print(states.shape, actions.shape)

# values of actions:
print(np.unique(actions))

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# create a Dataset object
class CustomDataset(Dataset):
    def __init__(self, states, actions, sequence_length):
        self.states = [torch.tensor(states[i:i+sequence_length]).float() for i in range(len(states) - sequence_length + 1)]
        self.actions = [torch.tensor(actions[i:i+sequence_length]).long() for i in range(len(actions) - sequence_length + 1)]
        self.sequence_length = sequence_length
        self.num_agents = min(1, actions.shape[1])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # random sample x from the number of agents
        x = np.random.randint(self.num_agents)
        return self.states[idx], self.actions[idx][:, x]
    
sequence_length = 8
dataset = CustomDataset(states, actions, sequence_length)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# main function

# Create an instance of the MLP
input_size = states.shape[1]
hidden_size = 64
output_size = np.unique(actions).shape[0]
mlp = MLP(input_size, hidden_size, output_size)
num_agents = dataset.num_agents

# Train the MLP using the generated data
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=1e-2)

num_epochs = 100
for epoch in range(num_epochs):
    batch_loss = []
    batch_accuracy = []
    for i, (state, action) in enumerate(dataloader):
        optimizer.zero_grad()
        output = mlp(state)
        loss = 0
        accuracy = 0
        agent_output = output.permute(0, 2, 1) # (batch_size, num_classes, sequence_length)
        agent_action = action[:, :]
        loss += criterion(agent_output, agent_action)
        accuracy += (torch.argmax(agent_output, dim=1) == agent_action).float().mean().item()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        accuracy = accuracy / num_agents
        batch_accuracy.append(accuracy)

    # print both the average and last epoch loss and accuracy
    print(f"Epoch {epoch+1}/{num_epochs}, Mean Loss: {np.mean(batch_loss):.4f}, Mean Accuracy: {np.mean(batch_accuracy):.4f}\
          , Last Loss: {batch_loss[-1]:.4f}, Last Accuracy: {batch_accuracy[-1]:.4f}")