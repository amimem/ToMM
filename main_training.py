from torch.utils.data import Dataset, DataLoader
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Sampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from AbstractionModelJointPolicy import AbstractionModelJointPolicy
from utils import MultiChannelNet
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--model_name', type=str,
                    default='STOMMnet_2_10', help='Name of the model')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--learning_rate', type=float,
                    default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--outdir', type=str, default='output/',
                    help='Output directory')
parser.add_argument('--data_filename', type=str,
                    default='_trainingdata_bitpop_exploit_True_numepi10000_K10_M2_N10_T10.npy', help='Data filename')
parser.add_argument('--seed', type=int, default=0, help='Random seed')

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device", flush=True)

seed: int = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class CustomDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


if __name__ == '__main__':

    # model and training setting
    model_name = args.model_name
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    # load the data in the output folder
    outdir = args.outdir
    data_filename = args.data_filename
    data = np.load(outdir + data_filename, allow_pickle=True).item()

    states = data["states"][0]
    actions = data["actions"][0]

    dataset = CustomDataset(states, actions)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # load system parameters
    action_space_dim = 2
    state_space_dim = data['sys_parameters']['K']
    num_agents = data['sys_parameters']['N']

    # instantiate model
    hidden_capacity = 250
    if model_name.split('_')[0] == 'STOMMnet':
        num_abs_agents = int(model_name.split('_')[1]) # for bitpop, could match data['sys_parameters']['jointagent_groundmodel_paras']['M']
        abs_action_space_dim = int(model_name.split('_')[2])
        assert (hidden_capacity/num_abs_agents).is_integer(
        ), "num of abstract agents should divide hidden dimensions"
        enc_hidden_dim = int(hidden_capacity/num_abs_agents)
        net = AbstractionModelJointPolicy(
            state_space_dim,
            abs_action_space_dim,
            enc_hidden_dim,
            num_agents,
            num_abs_agents,
            action_space_dim=action_space_dim
        )
    elif model_name == 'singletask_baseline':
        assert (hidden_capacity /
                num_agents).is_integer(), "num of agents should divide hidden dimensions"
        net = MultiChannelNet(
            n_channels=num_agents,
            input_size=state_space_dim,
            hidden_layer_width=int(hidden_capacity/num_agents),
            output_size=action_space_dim
        )
    elif model_name == 'multitask_baseline':
        net = MultiChannelNet(
            n_channels=1,
            input_size=state_space_dim,
            hidden_layer_width=hidden_capacity,
            output_size=num_agents*action_space_dim,
            output_dim=(num_agents, action_space_dim)
        )
    else:
        print('choose valid model')
    net.to(device)

    criterion = nn.CrossEntropyLoss()  # takes logits

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    logging_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            action_logit_vectors = net(inputs)
            loss = criterion(torch.swapaxes(
                action_logit_vectors, 1, 2), labels.type(torch.LongTensor))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader)}", flush=True)
        logging_loss.append(running_loss/len(train_loader))
    np.save(outdir + data_filename[:-4] + "_loggedloss.npy", logging_loss)
    torch.save(net.state_dict(), outdir +
               data_filename[:-4] + "_" + model_name +".pt")
