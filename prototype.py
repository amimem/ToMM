import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import logging

from Environment import Environment
from STOMPnet import STOMPnet

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)

writer = SummaryWriter()

# System parameters
K = 10   # states
L = 10   # abstract actions
M = 10   # abstract agents
N = 100  # agents

# Ground system
num_agents = N
state_space_dim = K  # vector space dimensionality
action_space_dim = 2 # number of discrete actions; fixed to 2 for simplicity
epsiode_length = 10

# Abstracted system
num_abs_agents = M
abs_action_space_dim = L # number of discrete abstract actions
#abstract action policy network parameters
enc_hidden_dim = 256

# Initialize environment and model
env = Environment(state_space_dim, num_agents, epsiode_length)

# Initialize model
model = STOMPnet(
    state_space_dim,
    abs_action_space_dim,
    enc_hidden_dim,
    num_agents,
    num_abs_agents,
    action_space_dim=action_space_dim
    )

if __name__ == '__main__':
    num_steps = 5
    episode_time_indices = []
    for step in range(num_steps):
        logging.info(f"Step: {step}")
        writer.add_graph(model, env.state)
        writer.close()
        logging.info(f"Input state: {env.state}")
        actions = model.forward(env.state)
        logging.info(f"Output actions: {actions}")
        env.state, episode_step = env.step(env.state, actions)
        episode_time_indices.append(episode_step)