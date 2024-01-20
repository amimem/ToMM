import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os

from Environment import Environment
from GroundModelJointPolicy import GroundModelJointPolicy


def generate_system_data(sys_parameters, sim_parameters, output_path):

    # assign system parameters
    state_space_dim = sys_parameters['K']   # state space dimension
    num_agents = sys_parameters['N']   # agents
    action_space_dim = sys_parameters['action_space_dim']

    # assign sim parameters
    epsiode_length = sim_parameters['episode_length']
    num_episodes = sim_parameters['num_episodes']
    exploit_mode = sim_parameters['exploit_mode']

    # assign data generation parameters
    seedlist = range(sim_parameters['num_seeds'])
    num_steps = sim_parameters['num_episodes']*epsiode_length

    # Groundmodel
    jointagent_groundmodel_paras = sys_parameters['jointagent_groundmodel_paras']
    groundmodel_name = jointagent_groundmodel_paras['groundmodel_name']

    # Initialize environment
    dummy_seed = 1
    env = Environment(state_space_dim, num_agents, epsiode_length, dummy_seed)

    # Initialize groundmodel
    num_agent_groups = jointagent_groundmodel_paras['M']
    agents_per_group = int(num_agents/num_agent_groups)
    model = GroundModelJointPolicy(
        state_space_dim,
        num_agent_groups,
        agents_per_group,
        action_space_dim=action_space_dim,
        model_paras=jointagent_groundmodel_paras
    )

    # run rollouts
    time_store = []
    state_store = []
    jointaction_store = []
    for seed in seedlist:
        episode_time_indices = []
        state_seq = []
        joint_action_seq = []
        env.state = env.sample_initial_state(state_space_dim, seed)
        for step in range(num_steps):

            state_seq.append(env.state.detach().cpu().numpy())
            action_probability_vectors = model.forward(env.state)

            if exploit_mode:  # take greedy action
                actions = torch.argmax(action_probability_vectors, dim=-1)
            else:  # sample
                # Categorical has batch functionality!
                actions = Categorial(action_probability_vectors)
            joint_action_seq.append(actions.detach().cpu().numpy())
            env.state, episode_step = env.forward(env.state, actions)
            episode_time_indices.append(episode_step)
        time_store.append(episode_time_indices)
        state_store.append(state_seq)
        jointaction_store.append(joint_action_seq)

    # add data to simulation parameter dictionary and store
    sim_parameters['sys_parameters'] = sys_parameters
    sim_parameters["times"] = np.array(time_store)
    sim_parameters["states"] = np.array(state_store)
    sim_parameters["actions"] = np.array(jointaction_store)
    filename = f"{output_path}_trainingdata_{groundmodel_name}_exploit_{exploit_mode}\
        _numepi{num_episodes}_K{state_space_dim}_M{num_agent_groups}_N{num_agents}_T{epsiode_length}.npy"

    with open(filename, 'wb') as f:
        np.save(f, sim_parameters)


if __name__ == '__main__':

    output_path = os.path.join(os.getcwd(), 'output/')

    sys_parameters = {}
    sys_parameters['K'] = 10  # state space dimension
    sys_parameters['N'] = 10  # agents
    # number of discrete actions; fixed to 2 for simplicity
    sys_parameters['action_space_dim'] = 2

    jointagent_groundmodel_paras = {}
    jointagent_groundmodel_paras['groundmodel_name'] = 'bitpop'
    jointagent_groundmodel_paras["corr"] = 0.8  # action pair correlation
    jointagent_groundmodel_paras['gen_type'] = 'sum'
    jointagent_groundmodel_paras['M'] = 2  # number of agent groups

    assert (sys_parameters['N']/jointagent_groundmodel_paras['M']).is_integer(), \
        "number of agents groups should divide total number of agents for some groundmodels"

    sys_parameters['jointagent_groundmodel_paras'] = jointagent_groundmodel_paras

    sim_parameters = {}
    sim_parameters['exploit_mode'] = True
    sim_parameters['episode_length'] = 10
    sim_parameters['num_episodes'] = 10000
    sim_parameters['num_seeds'] = 1

    generate_system_data(sys_parameters, sim_parameters, output_path)
