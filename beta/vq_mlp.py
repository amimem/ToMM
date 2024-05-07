import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import h5py
import yaml
import torch
import wandb
import time
import os
from utils import VectorQuantizer

# get slurm job array index
try:
    job_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
except:
    job_id = -1
    print("Not running on a cluster")
    exit()

seed = 0

# sets the seed for generating random numbers
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2, num_embeddings=64, embedding_dim=64, commitment_cost=0.25):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size) # input_size: sequence_length * (state_dim + num_actions)
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x, vq_loss = self.vq(x)
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x, vq_loss

# create a Dataset object
class CustomDataset(Dataset):
    def __init__(self, states, actions, sequence_length, seed=0, train=True):
        # get a rng
        self.rng = np.random.RandomState(seed)
        self.train = train
        self.states = [torch.tensor(states[i:i+sequence_length]).float() for i in range(len(states) - sequence_length + 1)]
        self.actions = [torch.tensor(actions[i:i+sequence_length]).long() for i in range(len(actions) - sequence_length + 1)]
        self.sequence_length = sequence_length
        self.num_agents = actions.shape[1]
        self.train_agents = self.rng.randint(0, self.num_agents, size=len(self.states))
        self.validation_agents = self.rng.randint(0, self.num_agents, size=len(self.states))
        for i, (t,v) in enumerate(zip(self.train_agents, self.validation_agents)):
            if t == v:
                self.validation_agents[i] = (v + 1) % self.num_agents

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # random sample x from the number of agents
        if self.train:
            split = self.train_agents
        else:
            split = self.validation_agents
        return self.states[idx], self.actions[idx][:, split[idx]]
    
def train(model, dataloader, num_actions=2):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

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
        
    return np.mean(batch_loss), np.mean(batch_accuracy)

def test(model, dataloader, num_actions=2):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    batch_loss = []
    batch_accuracy = []
    for i, (state, action) in enumerate(dataloader):
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
        batch_loss.append(loss.item())
        batch_accuracy.append(accuracy)
        
    return np.mean(batch_loss), np.mean(batch_accuracy)


def get_data_loader(data, sequence_length):
    states = data["states"]
    actions = data["actions"]
    # shape of states: (num_states, dim_teacher_inp)
    # shape of actions: (num_states, num_teachers)
    print(states.shape, actions.shape)

    train_dataset = CustomDataset(states, actions, sequence_length, seed=seed, train=True)
    test_dataset = CustomDataset(states, actions, sequence_length, seed=seed, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_dataloader, test_dataloader

def get_model(dataloader, num_actions, hidden_size=64, num_hidden_layers=2, num_epochs=10):

    state, action = next(iter(dataloader))
    # state shape: (batch_size, sequence_length, state_dim)
    # action shape: (batch_size, sequence_length)
    state = state.flatten(start_dim=1) # (batch_size, sequence_length * state_dim)
    action_onehot = torch.nn.functional.one_hot(action, num_classes=num_actions).flatten(start_dim=1) # (batch_size, sequence_length * num_actions)
    state = torch.hstack([state, action_onehot.float()]) # (batch_size, sequence_length * (state_dim + num_actions))

    # Create an instance of the MLP
    input_size = state.shape[1] # sequence_length * (state_dim + num_actions)
    hidden_size = hidden_size # hidden_size
    num_hidden_layers = num_hidden_layers # num_hidden_layers
    num_epochs = num_epochs

    mlp = MLP(input_size, hidden_size, num_actions, num_hidden_layers)      

    return mlp

if __name__ == "__main__":
    set_seed(seed)

    # for n_agents and n_groups = n_agents
    data_hashes = ["data_6013b64ce8", "data_dfdaecc3ee", "data_8646a4bdd8", "data_e94dbedcea",\
                    "data_813427ec72", "data_fbe4154fff", "data_d4fcf6cdef", "data_8950a6aae5",\
                    "data_97b6c3ab33", "data_d8fd9e8472", "data_fcb20c7d4a", "data_e8cbc57b61",\
                    "data_f714057b40", "data_567898bdec", "data_f2b68367cd", "data_d4588ac462"] + \
                   ["data_6013b64ce8", "data_27dea4bcce", "data_41f896f2be", "data_9bd5f5ee5f",\
                    "data_4af6f9d879", "data_a71679fd65", "data_46ef7fc2a7", "data_935be5ac8f",\
                    "data_eb9b7315c6", "data_0b7d25ce95", "data_89a277ffd9", "data_cf84771ef1",\
                    "data_76a571ea15", "data_dbf79f7d01", "data_5f734fb2a1", "data_6644bb7ada"]
    
    
    # data_hashes = ["data_6013b64ce8", "data_27dea4bcce", "data_41f896f2be", "data_9bd5f5ee5f",\
    #                 "data_4af6f9d879", "data_a71679fd65", "data_46ef7fc2a7", "data_935be5ac8f",\
    #                 "data_eb9b7315c6", "data_0b7d25ce95", "data_89a277ffd9", "data_cf84771ef1",\
    #                 "data_76a571ea15", "data_dbf79f7d01", "data_5f734fb2a1", "data_6644bb7ada"] # for n_agents and n_groups = 1
    
    # make sure all data_hashes are in the output folder
    assert all([os.path.exists(f"output/{data_hash}") for data_hash in data_hashes])
    print("All data hashes are in the output folder")

    sequence_lengths = [8]
    w_d = [(128, 2)]
    num_epochs = 20

    df = pd.DataFrame(columns=[
    "data_hash", 
    "num_agents", 
    "num_groups", 
    "state_dim", 
    "num_actions", 
    "sequence_length", 
    "hidden_size", 
    "num_hidden_layers", 
    "epoch", 
    "loss", 
    "accuracy"
    ])

    # sequence_lengths = [sequence_lengths[job_id]]
    # print(f"Running sequence length {sequence_lengths}")

    data_hashes = [data_hashes[job_id]]

    for data_hash in data_hashes:
        for sequence_length in sequence_lengths:
            for hidden_size, num_hidden_layers in w_d:
                data, config = load_data(data_hash)
                num_actions = config["file_attrs"]["num_actions"]
                config.update({"sequence_length": sequence_length, "hidden_size": hidden_size, "num_hidden_layers": num_hidden_layers})
                wandb.init(project="MLP", group="May_7th_seq_8", job_type=None, config=config)
                train_dataloader, test_dataloader = get_data_loader(data, sequence_length)
                mlp = get_model(train_dataloader, num_actions, hidden_size, num_hidden_layers)

                for epoch in range(num_epochs):
                    # train the model
                    train_epoch_loss, train_epoch_accuracy = train(mlp, train_dataloader, num_actions=num_actions)
                    # test the model
                    test_epoch_loss, test_epoch_accuracy = test(mlp, test_dataloader, num_actions=num_actions)
                    # print both the average and last epoch loss and accuracy
                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.4f},\
                          Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_accuracy:.4f}")
                    wandb.log({"epoch": epoch, "train_loss": train_epoch_loss, "train_accuracy": train_epoch_accuracy, \
                                "test_loss": test_epoch_loss, "test_accuracy": test_epoch_accuracy})
                    # Append the data to the DataFrame
                    new_row = pd.DataFrame({
                        "data_hash": [data_hash],
                        "num_agents": [config["file_attrs"]["num_teachers"]],
                        "num_groups": [config["file_attrs"]["num_groups"]],
                        "state_dim": [config["file_attrs"]["dim_state"]],
                        "num_actions": [num_actions],
                        "sequence_length": [sequence_length],
                        "hidden_size": [hidden_size],
                        "num_hidden_layers": [num_hidden_layers],
                        "epoch": [epoch],
                        "train_loss": [train_epoch_loss],
                        "train_accuracy": [train_epoch_accuracy],
                        "test_loss": [test_epoch_loss],
                        "test_accuracy": [test_epoch_accuracy]
                    }, index=[0])
                    df = pd.concat([df, new_row], ignore_index=True)
                wandb.finish()

    time_str = time.strftime("%Y-%m-%d-%H-%M")
    df.to_csv(f"output/{time_str}_results.csv", index=False)