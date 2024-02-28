from torch import nn
import torch
import numpy as np
from numpy import random
import itertools


class GroundModelJointPolicy():
    """
    A class representing a joint policy for ground-level agents in a multi-agent reinforcement learning setting.
    The observation is the index of the state space orthant (indexed in binary in state_set) containing the current state.

    Args:
        state_space_dim (int): The dimensionality of the state space.
        num_agent_groups (int): The number of abstract agents.
        agents_per_group (int): The number of ground-level agents per abstract agent.
        action_space_dim (int, optional): The dimensionality of the action space. Defaults to 2.
        model_paras (dict, optional): Additional model parameters. Defaults to None.

    Attributes:
        num_agents (int): The total number of ground-level agents.
        state_set (numpy.ndarray): An array containing all vertices of the unit hypercube.
        num_states (int): The number of states in the state set.
        action_space_dim (int): The dimensionality of the action space.
        action_policies (numpy.ndarray): An array representing the action policies for each ground-level agent.

    Methods:
        forward(state): Computes the action probability vectors for the given state.
        get_state_idx(state): Returns the index of the closest state in the state set.

    """

    def __init__(self, num_agents, state_space_dim, action_space_dim=2, model_paras=None):
        super(GroundModelJointPolicy, self).__init__()

        self.num_agents = num_agents
        self.state_set = np.array([np.array(l) for l in list(map(list, itertools.product(
            [0, 1], repeat=state_space_dim)))])  # all vertices of unit hypercube
        self.num_states = len(self.state_set)
        self.action_space_dim = action_space_dim

        rng = np.random.default_rng()
        self.action_policies = np.zeros(
            (self.num_agents, self.num_states), dtype=bool)

        if model_paras['modelname'] == 'bitpop':

            self.corr = model_paras['corr']
            gen_type = model_paras['ensemble']
            num_agent_groups = model_paras['M']
            assert (num_agents/num_agent_groups).is_integer(
            ), "Uneven group sizes. bitpop model requires group sizes to be equal"
            agents_per_group = int(num_agents/num_agent_groups)

            for agent_group_idx in range(num_agent_groups):
                agent_indices = range(agent_group_idx * agents_per_group,
                                      (agent_group_idx + 1) * agents_per_group)
                agent_indices_bool = np.zeros(self.num_agents, dtype=bool)
                agent_indices_bool[agent_indices] = True
                if (gen_type == "mix"):  # Bernoulli mixture of independent and identical binary RVs
                    is_same = self.corr > rng.random(self.num_states)
                    n_same = np.sum(is_same)
                    n_diff = self.num_states - n_same
                    self.action_policies[np.ix_(agent_indices_bool, is_same)] = rng.integers(
                        0, 2, n_same)[np.newaxis, :]
                    self.action_policies[np.ix_(agent_indices_bool, ~is_same)] = rng.integers(
                        0, 2, [agents_per_group, n_diff])
                elif (gen_type == "sum"):  # signed sum of independent and identical normal RVs
                    rho_normaldist = np.sin(np.pi / 2 * self.corr)
                    self.action_policies[agent_indices_bool, :] = (
                        np.sqrt(1 - rho_normaldist)
                        * rng.normal(size=(agents_per_group, self.num_states))
                        + np.sqrt(rho_normaldist)
                        * rng.normal(size=self.num_states)[np.newaxis, :]) > 0
                else:
                    print("choose sum or mix")
        else:
            print('use a defined groundmodel')

    def forward(self, state):
        """
        Computes the action probability vectors for the given state.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The action probability vectors.

        """
        state_idx = self.get_state_idx(state.detach().cpu().numpy())
        # the following is for actionspace_dim=2 only
        action_0_probabilities = self.action_policies[:, state_idx][:, None]
        action_probability_vectors = np.hstack(
            (action_0_probabilities, ~action_0_probabilities))
        return torch.Tensor(action_probability_vectors)

    def get_state_idx(self, state):
        """
        Returns the index of the closest state in the state set.

        Args:
            state (numpy.ndarray): The input state.

        Returns:
            int: The index of the closest state.

        """
        return np.argmin(np.linalg.norm(self.state_set-(state[None, :] > 0), axis=1))
