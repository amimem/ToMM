import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import logging

from Environment import Environment
from AbstractionModelJointPolicy import AbstractionModelJointPolicy

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
model = AbstractionModelJointPolicy(
    state_space_dim,
    abs_action_space_dim,
    enc_hidden_dim,
    num_agents,
    num_abs_agents,
    action_space_dim=action_space_dim
    )

if __name__ == '__main__':
    
    #example rollout
    num_steps = 5
    episode_time_indices = []
    for step in range(num_steps):
        logging.info(f"Step: {step}")
        writer.add_graph(model, env.state)
        writer.close()
        logging.info(f"Input state: {env.state}")
        action_probability_vectors = model.forward(env.state)
        
        #take greedy action
        actions = torch.argmax(action_probability_vectors,dim=-1)
        #sample
        # actions=[choices(action_probability_vector) for action_probability_vector in action_probability_vectors]  #from random import choices # Pyver>=3.6
        logging.info(f"Output actions: {actions}")
        env.state, episode_step = env.step(env.state, actions)
        episode_time_indices.append(episode_step)


