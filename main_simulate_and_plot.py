import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os

from Environment import Environment
from STOMPnet import STOMPnet
from GroundModelJointPolicy import GroundModelJointPolicy
from utils import compare_plot

# System parameters
K = 10   # state space dimension
L = 10   # abstract actions
M = 2   # abstract agents
N = 10  # agents

assert int(N/M) == N / \
    M, "number of abstract agents should divide ground agents for some groundmodels"

# Ground system
num_agents = N
state_space_dim = K  # vector space dimensionality
action_space_dim = 2  # number of discrete actions; fixed to 2 for simplicity
epsiode_length = 10

# Abstracted system
num_abs_agents = M
abs_action_space_dim = L  # number of discrete abstract actions
# abstract action policy network parameters
enc_hidden_dim = 256

# Initialize environment
seed = 1
env = Environment(state_space_dim, num_agents, epsiode_length, seed)

# Initialize abstraction system model
abstractionmodel = STOMPnet(
    state_space_dim,
    abs_action_space_dim,
    enc_hidden_dim,
    num_agents,
    num_abs_agents,
    action_space_dim=action_space_dim
)

# Initialize ground system model
baseline_paras = {}
baseline_name = 'bitpop'
# ...other baseline names here
baseline_paras['baseline_name'] = baseline_name

if baseline_name == 'bitpop':
    baseline_paras["corr"] = 0.8
    # baseline_paras["acrosscorr"] = 0
    # baseline_paras['gen_type'] = 'mix'
    baseline_paras['gen_type'] = 'sum'
else:
    ...#other baselines parameters here

agents_per_abstract_agent = int(num_agents/num_abs_agents)
groundmodel = GroundModelJointPolicy(
    state_space_dim,
    num_abs_agents,
    agents_per_abstract_agent,
    action_space_dim=2,
    baseline_paras=baseline_paras
)

if __name__ == '__main__':
    output_path = 'output/'
    # if ~os.path.exists(output_path):
    #     os.makedirs(output_path)

    # example rollout
    num_episodes = 100
    num_steps = num_episodes*epsiode_length
    exploit_mode = True
    output_filenames = []
    for model_name, model in zip(['groundmodel', 'abstractionmodel'], [groundmodel, abstractionmodel]):
        episode_time_indices = []
        state_seq = []
        joint_action_seq = []
        print(f'seed:{seed}')
        env.state = env.sample_initial_state(state_space_dim, seed)
        for step in range(num_steps):

            state_seq.append(env.state.detach().cpu().numpy())
            action_probability_vectors = model.forward(env.state)

            if exploit_mode:  # take greedy action
                actions = torch.argmax(action_probability_vectors, dim=-1)
            else:  # sample
                # Categorical has batch functionality!
                actions = Categorical(action_probability_vectors)
            joint_action_seq.append(actions.detach().cpu().numpy())
            env.state, episode_step = env.step(env.state, actions)
            episode_time_indices.append(episode_step)

        data = {}
        data['model_name'] = model_name
        data['exploit_mode'] = exploit_mode
        data['K'] = K
        data['L'] = L
        data['M'] = M
        data['N'] = N
        data['T'] = epsiode_length
        data["times"] = episode_time_indices
        data["states"] = state_seq
        data["actions"] = joint_action_seq
        if model_name == 'groundmodel':
            data['baseline_paras'] = baseline_paras
        filename = f"output/rundata_{model_name}_exploit{exploit_mode}_{K}_L{L}_M{M}_N{N}_T{epsiode_length}.npy"
        np.save(filename, data)
        output_filenames.append(filename)

    # post simulation analysis
    compare_plot(output_filenames)
