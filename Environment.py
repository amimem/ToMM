import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Environment:
    def __init__(self, state_space_dim, num_agents, episode_length, fluctuation_strength_factor=3, start_seed=1):
        """
        Initializes an RNN environment on which multiple agents act.

        Args:
            state_space_dim (int): Dimension of the state space.
            num_agents (int): Number of agents in the environment.
            episode_length (int): Length of each episode.
            fluctuation_strength_factor (float): "g" (control parameter for stability transition). critical value g_{crit}~order(1). exact value of g_{crit} depends on nonlinearity and weight ensemble.
            start_seed (int, optional): Starting seed for random number generation. Defaults to 1.
        """
        self.state_space_dim = state_space_dim
        self.RNNdim = 100
        assert self.state_space_dim <= self.RNNdim, "can observe more degrees of freedom than are in system"
        # episodic properties
        self.T = episode_length
        self.counter = 0
        self.seed = start_seed
        # Initialize the state and transition function
        self.state = self.sample_initial_state(self.seed)

        # instantiate rnn
        input_size = num_agents
        hidden_size = self.RNNdim
        self.RNN = torch.nn.RNN(input_size, hidden_size, bias=False)
        for param in self.RNN.parameters():
            param.requires_grad = False
        input_variance_weight = 0.5 / # input-to-recurrent variance ratio
        input_dilution_factor = 0.5  # average action value (uniform on {0,1})
        nn.init.normal_(self.RNN.weight_ih_l0, std=fluctuation_strength_factor *
                        np.sqrt(input_variance_weight/(input_dilution_factor*input_size)))
        nn.init.normal_(self.RNN.weight_hh_l0, std=fluctuation_strength_factor *
                        np.sqrt((1-input_variance_weight)/hidden_size))
        # nn.init.orthogonal_(self.RNN.weight_ih_l0, gain=fluctuation_strength_factor*np.sqrt(input_variance_weight/(input_dilution_factor*input_size)))
        # nn.init.orthogonal_(self.RNN.weight_hh_l0, gain=fluctuation_strength_factor*np.sqrt((1-input_variance_weight)/hidden_size))

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
            return self.sample_initial_state(seed=self.seed), self.T
        else:
            _, next_state = self.RNN(torch.unsqueeze(
                actions.to(torch.float32), 0), torch.unsqueeze(state, 0))
            return torch.squeeze(next_state), self.counter

    def sample_initial_state(self, seed=0):
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
        # Sample an initial state from a squashed Gaussian distribution
        return F.tanh(torch.Tensor(np.random.randn(self.RNNdim)))
