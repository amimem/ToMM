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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device", flush=True)

seed: int = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

if __name__ == '__main__':

	epochs = 10
	learning_rate = 1e-3
	batch_size = 16

	# load the data in the output folder
	# outdir = "/Users/ens/repos/marl/output/"
	outdir = "output/"
	data = np.load(outdir + "_trainingdata_groundmodel_exploit_True_numepi10000_K10_L10_M2_N10_T10.npy", allow_pickle=True).item()

	states = data["states"][0]
	actions = data["actions"][0]

	dataset = CustomDataset(states, actions)
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


	# Abstraction system
	num_abs_agents = data['sys_parameters']['M']
	abs_action_space_dim = data['sys_parameters']['L']  # number of discrete abstract actions
	# abstract action policy network parameters
	enc_hidden_dim = 256
	action_space_dim = 2
	state_space_dim = data['sys_parameters']['K']
	num_agents = data['sys_parameters']['N']
	# Initialize abstraction system model
	net = AbstractionModelJointPolicy(
	    state_space_dim,
	    abs_action_space_dim,
	    enc_hidden_dim,
	    num_agents,
	    num_abs_agents,
	    action_space_dim=action_space_dim
	)
	net.to(device)

	# criterion = nn.CrossEntropyLoss() takes logits
	criterion = nn.NLLLoss() #takes log probabilities

	optimizer = optim.Adam(net.parameters(), lr=learning_rate)

	logging_loss = [] 
	for epoch in range(epochs):
	    running_loss = 0.0
	    for i, data in enumerate(train_loader, 0):
	        inputs, labels = data
	        inputs = inputs.to(device)
	        labels = labels.float().to(device)
	        optimizer.zero_grad()
	        action_probability_vectors = net(inputs)
	        loss = criterion(torch.swapaxes(torch.log(action_probability_vectors),1,2), labels.type(torch.LongTensor))
	        loss.backward()
	        optimizer.step()
	        running_loss += loss.item()
	    print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader)}", flush=True)
	    logging_loss.append(running_loss/len(train_loader))
	torch.save(net.state_dict(), outdir + "groundmodel.pt")