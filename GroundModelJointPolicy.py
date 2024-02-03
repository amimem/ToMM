from torch import nn
import torch
import numpy as np
from numpy import random
import itertools


class GroundModelJointPolicy(nn.Module):
    def __init__(self, state_space_dim, num_abs_agents, agents_per_abstract_agent, action_space_dim=2, model_paras=None):
        super(GroundModelJointPolicy, self).__init__()
        self.num_agents = num_abs_agents*agents_per_abstract_agent
        self.state_set = np.array([np.array(l) for l in list(map(list, itertools.product(
            [0, 1], repeat=state_space_dim)))])  # all vertices of unit hypercube
        self.num_states = len(self.state_set)
        self.action_space_dim = action_space_dim

        rng = np.random.default_rng()
        self.action_policies = np.zeros(
            (self.num_agents, self.num_states), dtype=bool)
        agent_indices_bool = np.zeros(self.num_agents, dtype=bool)
        
        if model_paras['groundmodel_name'] == 'bitpop':
            self.corr = model_paras['corr']
            gen_type = model_paras['gen_type']
            for abs_agent_idx in range(num_abs_agents):
                agent_indices = range(abs_agent_idx * agents_per_abstract_agent,
                                      (abs_agent_idx + 1) * agents_per_abstract_agent)
                agent_indices_bool[agent_indices] = True
                if (gen_type == "mix"):  # Bernoulli mixture of independent and identical binary RVs
                    is_same = self.corr > rng.random(self.num_states)
                    n_same = np.sum(is_same)
                    n_diff = self.num_states - n_same
                    self.action_policies[np.ix_(agent_indices_bool, is_same)] = rng.integers(
                        0, 2, n_same)[np.newaxis, :]
                    self.action_policies[np.ix_(agent_indices_bool, ~is_same)] = rng.integers(
                        0, 2, [agents_per_abstract_agent, n_diff])
                elif (gen_type == "sum"):  # signed sum of independent and identical normal RVs
                    rho_normaldist = np.sin(np.pi / 2 * self.corr)
                    self.action_policies[agent_indices_bool, :] = (
                        np.sqrt(1 - rho_normaldist)
                        * rng.normal(size=(agents_per_abstract_agent, self.num_states))
                        + np.sqrt(rho_normaldist)
                        * rng.normal(size=self.num_states)[np.newaxis, :]) > 0
                else:
                    print("choose sum or mix")
                agent_indices_bool[agent_indices] = False
        else:
            print('use a defined groundmodel')

    def forward(self, state):
        state_idx = self.get_state_idx(state.detach().cpu().numpy())
        action_probability_vectors = np.hstack(
            (self.action_policies[:, state_idx][:, None], ~self.action_policies[:, state_idx][:, None]))
        return torch.Tensor(action_probability_vectors)

    def get_state_idx(self, state):
        return np.argmin(np.linalg.norm(self.state_set-state[None, :], axis=1))
