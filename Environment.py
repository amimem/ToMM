import torch
import torch.nn as nn
import numpy as np

from utils import get_linear_nonlinear_function

def sample_initial_state(state_space_dim):
    # Sample an initial state from a uniform distribution
    return torch.Tensor(np.random.rand(state_space_dim))

class Environment(nn.Module):
    def __init__(self, state_space_dim, num_agents, episode_length):
        super(Environment, self).__init__()

        self.state_space_dim = state_space_dim

        #episodic properties
        self.T = episode_length
        self.counter = 0

        # Initialize the state and transition function
        self.state = sample_initial_state(state_space_dim)
        self.transition_func = get_linear_nonlinear_function(state_space_dim + num_agents, state_space_dim)

    def step(self, state, actions):
        # Apply the transition function to the state and actions
        self.counter += 1
        if self.counter == self.T: #reset episode
            self.counter = 0
            return sample_initial_state(self.state_space_dim), self.T
        else:
            return self.transition_func(torch.cat([state, actions])), self.counter