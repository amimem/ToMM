import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import h5py
import yaml
import torch

data_dir = "output/data_b4c7f51405"
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
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# create a Dataset object
class CustomDataset(Dataset):
    def __init__(self, states, actions):
        # make them float tensors
        self.states = torch.tensor(states).float()
        self.actions = torch.tensor(actions).squeeze().long()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
    
dataset = CustomDataset(states, actions)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# main function

# Create an instance of the MLP
input_size = states.shape[1]
hidden_size = 16
output_size = np.unique(actions).shape[0]
mlp = MLP(input_size, hidden_size, output_size)

# Train the MLP using the generated data
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    batch_loss = []
    batch_accuracy = []
    for i, (state, action) in enumerate(dataloader):
        optimizer.zero_grad()
        output = mlp(state)
        loss = criterion(output, action)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        accuracy = (torch.argmax(output, dim=1) == action).float().mean().item()
        batch_accuracy.append(accuracy)

    print(f"Epoch {epoch+1}, Loss: {np.mean(batch_loss)}, Accuracy: {np.mean(batch_accuracy)}")
