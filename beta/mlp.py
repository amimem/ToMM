import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import h5py
import yaml
import torch

def load_data(data_hash, data_seed=0):
    data_dir = f"output/{data_hash}"
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

    dataset = datasets[f"dataset_{data_seed}"]
    return dataset, config

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ) for _ in range(num_hidden_layers)],
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
        self.num_agents = actions.shape[1]
        # self.num_agents = min(2, actions.shape[1])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # random sample x from the number of agents
        x = np.random.randint(self.num_agents)
        return self.states[idx], self.actions[idx][:, x]
    
def train(model, dataloader, num_epochs=10, num_actions=2):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    epoch_loss = []
    epoch_accuracy = []

    for epoch in range(num_epochs):
        batch_loss = []
        batch_accuracy = []
        for i, (state, action) in enumerate(dataloader):
            optimizer.zero_grad()
            # bsz, seqlen, statedim = state.shape
            # bsz, seqlen = action.shape
            state = state.flatten(start_dim=1)
            action_onehot = torch.nn.functional.one_hot(action, num_classes=num_actions).flatten(start_dim=1)
            action_onehot[:, -num_actions:] = 0
            state = torch.hstack([state, action_onehot.float()])

            output = model(state)
            agent_output = output
            agent_action = action[:, -1]
            loss = criterion(agent_output, agent_action)
            accuracy = (torch.argmax(agent_output, dim=1) == agent_action).float().mean().item()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            batch_accuracy.append(accuracy)

        epoch_loss.append(np.mean(batch_loss))
        epoch_accuracy.append(np.mean(batch_accuracy))

        # print both the average and last epoch loss and accuracy
        print(f"Epoch {epoch+1}/{num_epochs}, Mean Loss: {np.mean(batch_loss):.4f}, Mean Accuracy: {np.mean(batch_accuracy):.4f}\
            , Last Loss: {batch_loss[-1]:.4f}, Last Accuracy: {batch_accuracy[-1]:.4f}")
        
    return epoch_loss, epoch_accuracy

def sweep(data_hash, sequence_length=16, hidden_size=64, num_hidden_layers=2, num_epochs=10):
    data_hash = data_hash
    data_seed = 0

    data, config = load_data(data_hash, data_seed)

    states = data["states"]
    actions = data["actions"]
    num_actions = config["file_attrs"]["num_actions"]

    # shape of states: (num_states, dim_teacher_inp)
    # shape of actions: (num_states, num_teachers)
    print(states.shape, actions.shape)

    sequence_length = 16
    dataset = CustomDataset(states, actions, sequence_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    num_agents = dataset.num_agents

    state, action = next(iter(dataloader))
    state = state.flatten(start_dim=1)
    action_onehot = torch.nn.functional.one_hot(action, num_classes=num_actions).flatten(start_dim=1)
    state = torch.hstack([state, action_onehot.float()])

    # Create an instance of the MLP
    input_size = state.shape[1]
    hidden_size = 64
    num_hidden_layers = 2
    num_epochs = num_epochs

    mlp = MLP(input_size, hidden_size, num_actions, num_hidden_layers)      

    return mlp, dataloader, num_actions 

if __name__ == "__main__":
    data_hashes = ["data_d4588ac462", "data_50d7e14370", "data_076e93c9c2"]
    sequence_lengths = [4, 8, 16, 32]
    w_d = [(32, 1), (64, 2), (128, 4), (256, 8)]
    
    for data_hash in data_hashes:
        for sequence_length in sequence_lengths:
            for hidden_size, num_hidden_layers in w_d:
                mlp, dataloader, num_actions = sweep(data_hash, sequence_length, hidden_size, num_hidden_layers)
                epoch_loss, epoch_accuracy = train(mlp, dataloader, num_epochs=10, num_actions=num_actions)
