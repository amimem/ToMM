import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os

from Environment import Environment
from GroundModelJointPolicy import GroundModelJointPolicy


def generate_system_data(sys_parameters, sim_parameters, output_path, dataset_label):

    # assign system parameters
    state_space_dim = sys_parameters['K']   # state space dimension
    num_agents = sys_parameters['N']   # agents
    action_space_dim = sys_parameters['action_space_dim']

    # assign sim parameters
    epsiode_length = sim_parameters['episode_length']
    num_episodes = sim_parameters['num_episodes']
    action_selection = sim_parameters['actsel']

    output_filename = f"{output_path}_{dataset_label}_simulationdata_actsel_{action_selection}_numepi_{num_episodes}_K_{state_space_dim}_N_{num_agents}_T_{epsiode_length}"
    sim_parameters["dataset_label"] = dataset_label
    sim_parameters['sys_parameters'] = sys_parameters
    np.save(output_filename+'.npy', sim_parameters)

    # assign data generation parameters
    seedlist = range(sim_parameters['num_seeds'])
    num_steps = sim_parameters['num_episodes']*epsiode_length

    # Initialize Groundmodel
    model = GroundModelJointPolicy(
        num_agents,
        state_space_dim,
        action_space_dim=action_space_dim,
        model_paras=sys_parameters['jointagent_groundmodel_paras']
    )

    # Initialize environment
    dummy_seed = 1
    env = Environment(state_space_dim, num_agents, epsiode_length, dummy_seed)

    # rollout model into a dataset of trajectories
    for seed in seedlist:
        print(f"running seed {seed} of {len(seedlist)}")
        episode_time_indices = []
        state_seq = []
        joint_action_seq = []
        env.state = env.sample_initial_state(state_space_dim, seed)
        for step in range(num_steps):

            state_seq.append(env.state.detach().cpu().numpy())
            action_probability_vectors = model.forward(env.state)

            if action_selection == 'greedy':  # take greedy action
                actions = torch.argmax(action_probability_vectors, dim=-1)
            else:  # sample
                # Categorical has batch functionality!
                actions = Categorical(action_probability_vectors)
            joint_action_seq.append(actions.detach().cpu().numpy())
            env.state, episode_step = env.step(env.state, actions)
            episode_time_indices.append(episode_step)
        sim_data = {}
        sim_data["times"] = np.array(episode_time_indices)
        sim_data["states"] = np.array(state_seq)
        sim_data["actions"] = np.array(joint_action_seq)
        np.save(output_filename+f'_dataseed_{seed}.npy', sim_data)


if __name__ == '__main__':

    output_path = os.path.join(os.getcwd(), 'output/')

    dataset_label = "3agentdebug"
    groundmodel_name = "bitpop"

    sys_parameters = {}
    sys_parameters['N'] = 4  # agents
    sys_parameters['K'] = 5  # state space dimension
    # number of discrete actions; fixed to 2 for simplicity
    sys_parameters['action_space_dim'] = 2

    jointagent_groundmodel_paras = {}
    jointagent_groundmodel_paras['modelname'] = groundmodel_name

    if groundmodel_name == "bitpop":
        jointagent_groundmodel_paras["corr"] = 1  # action pair correlation
        jointagent_groundmodel_paras['ensemble'] = 'sum'
        jointagent_groundmodel_paras['M'] = 2  # number of agent groups
        assert (sys_parameters['N']/jointagent_groundmodel_paras['M']).is_integer(), \
            "number of agents groups should divide total number of agents for some groundmodels"

    else:
        abort("select an implemented groundmodel")
    sys_parameters['jointagent_groundmodel_paras'] = jointagent_groundmodel_paras

    sim_parameters = {}
    sim_parameters['actsel'] = 'greedy'
    sim_parameters['episode_length'] = 100
    sim_parameters['num_episodes'] = 100
    sim_parameters['num_seeds'] = 2

    # add parameter setting of ground model to label
    dataset_label += ''.join(['_'+key+'_'+str(value)
                              for key, value in jointagent_groundmodel_paras.items()])

    generate_system_data(sys_parameters, sim_parameters,
                         output_path, dataset_label)
