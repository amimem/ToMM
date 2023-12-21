import torch
import torch.nn as nn
import numpy as np
import random

from utils import get_linear_nonlinear_function


class Environment(nn.Module):
    def __init__(self, state_space_dim, num_agents, episode_length, start_seed=1):
        super(Environment, self).__init__()

        self.state_space_dim = state_space_dim

        # episodic properties
        self.T = episode_length
        self.counter = 0
        self.seed = start_seed
        # Initialize the state and transition function
        self.state = self.sample_initial_state(state_space_dim, self.seed)
        self.transition_func = get_linear_nonlinear_function(
            state_space_dim + num_agents, state_space_dim)

    def step(self, state, actions):
        # Apply the transition function to the state and actions
        self.counter += 1
        if self.counter == self.T:  # reset episode
            self.counter = 0
            self.seed += 1
            return self.sample_initial_state(self.state_space_dim, self.seed), self.T
        else:
            return self.transition_func(torch.cat([state, actions])), self.counter

    def sample_initial_state(self, state_space_dim, seed):
        random.seed(seed)
        # Sample an initial state from a uniform distribution
        return torch.Tensor(np.random.rand(state_space_dim))
