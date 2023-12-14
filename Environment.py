import torch
import torch.nn as nn
import numpy as np

from gumbel_partition.utils import get_linearnonlinear_function

def sample_initial_state(state_dim):
    # Sample an initial state from a uniform distribution
    return torch.Tensor(np.random.rand(state_dim))

class Environment(nn.Module):
    def __init__(self, state_dim, num_agents):
        super(Environment, self).__init__()

        # Initialize the state and transition function
        self.state = sample_initial_state(state_dim)
        self.transition_func = get_linearnonlinear_function(state_dim + num_agents, state_dim)

    def step(self, state, actions):
        # Apply the transition function to the state and actions
        return self.transition_func(torch.vstack((state, torch.Tensor(actions))))