import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Environment(nn.Module):
    def __init__(self, state_space_dim, num_agents, episode_length, start_seed=1):
        """
        Initializes an environment for multi-agent reinforcement learning.

        Args:
            state_space_dim (int): Dimension of the state space.
            num_agents (int): Number of agents in the environment.
            episode_length (int): Length of each episode.
            start_seed (int, optional): Starting seed for random number generation. Defaults to 1.
        """
        super(Environment, self).__init__()

        self.state_space_dim = state_space_dim

        # episodic properties
        self.T = episode_length
        self.counter = 0
        self.seed = start_seed
        # Initialize the state and transition function
        self.state = self.sample_initial_state(state_space_dim, self.seed)

        # Create a linear layer
        self.linear_layer = nn.Linear(
            state_space_dim + num_agents, state_space_dim)
        stability_factor = 2
        nn.init.normal_(self.linear_layer.weight,std=stability_factor/np.sqrt(state_space_dim + num_agents))

    def forward(self, state, actions):
        """
        Performs a forward pass through the environment.

        Args:
            state (torch.Tensor): Current state.
            actions (torch.Tensor): Actions taken by the agents.

        Returns:
            tuple: A tuple containing the next state and the counter value.
        """
        # Apply the transition function to the state and actions
        self.counter += 1
        if self.counter == self.T:  # reset episode
            self.counter = 0
            self.seed += 1
            return self.sample_initial_state(self.state_space_dim, self.seed), self.T
        else:
            return F.sigmoid(self.linear_layer(torch.cat([state, actions]))), self.counter

    def sample_initial_state(self, state_space_dim, seed):
        """
        Samples an initial state from a uniform distribution.

        Args:
            state_space_dim (int): Dimension of the state space.
            seed (int): Seed for random number generation.

        Returns:
            torch.Tensor: Initial state.
        """
        # make sure the random seed set here does not interfere with the random seed set in the main_training.py file
        random.seed(seed)
        # Sample an initial state from a uniform distribution
        return torch.Tensor(np.random.rand(state_space_dim))
