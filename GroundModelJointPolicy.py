from torch import nn
import torch
import numpy as np
import itertools
from utils import get_untrained_net, get_width
import os
import yaml
def binary2index(var):
    if var.ndim == 1:
        return np.sum(np.power(2, np.arange(len(var))) * var).astype(int)
    elif var.ndim == 2:
        return np.sum(
            np.power(2, np.arange(var.shape[0]))[:, np.newaxis] * var, axis=0
        ).astype(int)
    else:
        print("why more than 2 dimensions?")

class GroundModelJointPolicy:
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

    def __init__(self, num_agents, state_space_dim, action_space_dim=2):
        super(GroundModelJointPolicy, self).__init__()

        self.num_agents = num_agents
        self.state_set = np.array([np.array(l) for l in list(map(list, itertools.product(
            [0, 1], repeat=state_space_dim)))],dtype=bool)  # all vertices of unit hypercube
        self.state_set_inds = [binary2index(var) for var in self.state_set]
        self.state_set = self.state_set[np.argsort(self.state_set_inds)]
        self.state_set_inds = list(np.sort(self.state_set_inds))
        self.num_states = len(self.state_set)
        self.action_space_dim = action_space_dim
        self.action_probability_vectors=np.zeros(
            (self.num_agents, self.num_states,self.action_space_dim), dtype=bool)
        self.preferred_actions = np.ones((self.num_agents,),dtype=bool)

    def forward(self, state):
        """
        Computes the action probability vectors for the given state.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The action probability vectors.

        """
        state_idx = self.get_state_idx(state)
        # the following is for actionspace_dim=2 and deterministic policies
        # action_0_probabilities = self.action_policies[:, state_idx][:, None]
        # action_probability_vectors = np.hstack(
        #     (action_0_probabilities, ~action_0_probabilities))
        return torch.Tensor(self.action_probability_vectors[state_idx])

    def get_state_idx(self, state):
        """
        Returns the index of the closest state in the state set.

        Args:
            state (numpy.ndarray): The input state.

        Returns:
            int: The index of the closest state.

        """
        return self.state_set_inds.index(binary2index((np.array(state[0]) > 0)))
    
    def set_action_policies(self, policy_params):
        """
        Set the action policies based on the given policy parameters.

        Args:
            policy_params (dict): A dictionary containing the policy parameters.
                model_name (str): The name of the model.
                seed (int): The random seed.
                corr (float): The correlation coefficient.
                ensemble (str): The ensemble type.
                M (int): The number of abstract agents.
                
        Raises:
            AssertionError: If the group sizes are uneven for the 'bitpop' model.

        Notes:
            - The 'bitpop' model requires group sizes to be equal.
            - The 'gen_type' parameter can be either 'mix' or 'sum'.
                - 'mix': Bernoulli mixture of independent and identical binary random variables.
                - 'sum': Signed sum of independent and identical normal random variables.
            - The action policies are updated based on the given policy parameters.

        Returns:
            None
        """

        if policy_params['ground_model_name'] == 'bitpop':
            corr = policy_params['corr']
            gen_type = policy_params['ensemble']
            num_agent_groups = policy_params['M']
            assert (self.num_agents/num_agent_groups).is_integer(), "Uneven group sizes. bitpop model requires group sizes to be equal"
            agents_per_group = int(self.num_agents/num_agent_groups)
            rng=policy_params['rng']
            action_policies = np.zeros(
            (self.num_agents, self.num_states), dtype=bool)
            for agent_group_idx in range(num_agent_groups):
                agent_indices = range(agent_group_idx * agents_per_group, (agent_group_idx + 1) * agents_per_group)
                agent_indices_bool = np.zeros(self.num_agents, dtype=bool)
                agent_indices_bool[agent_indices] = True
                if (gen_type == "mix"):  # Bernoulli mixture of independent and identical binary RVs
                    is_same = corr > rng.random(self.num_states)
                    n_same = np.sum(is_same)
                    n_diff = self.num_states - n_same
                    action_policies[np.ix_(agent_indices_bool, is_same)] = rng.integers(0, 2, n_same)[np.newaxis, :]
                    action_policies[np.ix_(agent_indices_bool, ~is_same)] = rng.integers(0, 2, [agents_per_group, n_diff])
                elif (gen_type == "sum"):  # signed sum of independent and identical normal RVs
                    rho_normaldist = np.sin(np.pi / 2 * corr)
                    action_policies[agent_indices_bool, :] = (np.sqrt(1 - rho_normaldist) * rng.normal(size=(agents_per_group, self.num_states)) +
                                                                    np.sqrt(rho_normaldist) * rng.normal(size=self.num_states)[np.newaxis, :]) > 0
                else:
                    print("choose sum or mix")
            if True:        
                self.preferred_actions = rng.integers(0, 2, self.num_agents)
                # Interpret existing action space as whether or not preferred action is taken.
                # To transform to action taken: if preferred is 0, then flip value, else leave unchanged
                # action_policies[~self.preferred_actions,:] = ~action_policies[~self.preferred_actions,:]
                action_policies = action_policies==self.preferred_actions[:,np.newaxis]
            
            action_0_probabilities= (action_policies.T==0)[:, :, None]
            self.action_probability_vectors = np.concatenate(
            (action_0_probabilities, ~action_0_probabilities),axis=-1)

        elif '/' in policy_params['ground_model_name']:
            #initialize model
            model=get_untrained_net(
                policy_params['origtraining_args'],
                policy_params['data_settings'], 
                policy_params['origtraining_args']["hidden_dim"], 
                policy_params['origtraining_args']['model_name']
                )
            model.load_state_dict(torch.load(policy_params["loadedmodel_path"]+"/state_dict_final.pt"))

            inputs = torch.from_numpy(self.state_set.astype(np.float32))
            self.action_probability_vectors = torch.nn.functional.softmax(model(inputs),dim=-1).detach().numpy()
        else:
            print('use a defined ground model')
