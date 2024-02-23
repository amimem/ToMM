import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Environment:
    def __init__(self, state_space_dim, num_agents, episode_length, start_seed=1):
        """
        Initializes an environment for multi-agent reinforcement learning.

        Args:
            state_space_dim (int): Dimension of the state space.
            num_agents (int): Number of agents in the environment.
            episode_length (int): Length of each episode.
            start_seed (int, optional): Starting seed for random number generation. Defaults to 1.
        """
        self.state_space_dim = state_space_dim

        # episodic properties
        self.T = episode_length
        self.counter = 0
        self.seed = start_seed
        # Initialize the state and transition function
        self.state = self.sample_initial_state(state_space_dim, self.seed)

        # Create a linear layer
        self.linear_layer = nn.Linear(
            # state_space_dim, state_space_dim)
            state_space_dim + num_agents, state_space_dim)
        for param in self.linear_layer.parameters():
            param.requires_grad = False
        stability_factor = 5 # coupling strength parameter g, sqrt(2) is critical value for tanh 
        # nn.init.normal_(self.linear_layer.weight, std=stability_factor / np.sqrt(2*state_space_dim + num_agents))
        nn.init.normal_(self.linear_layer.weight, std=stability_factor / np.sqrt(state_space_dim))

        # nn.init.xavier_normal_(self.linear_layer.weight, gain=stability_factor / np.sqrt(state_space_dim + num_agents))
        self.linear_layer.bias.fill_(0.)

        # matrix implementation
        # self.weight_matrix = torch.randn(state_space_dim, state_space_dim + num_agents)
        # stability_factor = 2
        # self.weight_matrix *= stability_factor / np.sqrt(state_space_dim + num_agents)

    def step(self, state, actions):
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
            # return self.linear_layer(F.tanh(state)), self.counter
            return F.tanh(self.linear_layer(torch.cat([state, actions]))), self.counter
            # return F.sigmoid(self.linear_layer(torch.cat([state, actions]))), self.counter

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
