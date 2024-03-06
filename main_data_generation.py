import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
import argparse
from Environment import Environment
from GroundModelJointPolicy import GroundModelJointPolicy
import time

parser = argparse.ArgumentParser(description='data generation parameters')
parser.add_argument('--stablefac', type=float,
                    default=8.0, help='stability factor')
parser.add_argument('--T', type=int,
                    default=1000, help='episode length')
parser.add_argument('--corr', type=float,
                    default=1.0, help='action correlation')
parser.add_argument('--N', type=int,
                    default=4, help='number of agents')
parser.add_argument('--M', type=int,
                    default=4, help='number of agent groups')
args = parser.parse_args()


def generate_system_data(sys_parameters, sim_parameters, output_path, dataset_label):

    # assign system parameters
    state_space_dim = sys_parameters['K']   # state space dimension
    num_agents = sys_parameters['N']   # agents
    fluctuation_strength_factor = sys_parameters['fluctuation_strength_factor']
    action_space_dim = sys_parameters['action_space_dim']

    # assign sim parameters
    episode_length = sim_parameters['episode_length']
    num_episodes = sim_parameters['num_episodes']
    action_selection = sim_parameters['actsel']

    output_filename = f"{output_path}_{dataset_label}_simulationdata_actsel_{action_selection}_numepi_{num_episodes}_K_{state_space_dim}_N_{num_agents}_T_{episode_length}_g_{fluctuation_strength_factor}"
    sim_parameters["dataset_label"] = dataset_label
    sim_parameters['sys_parameters'] = sys_parameters
    np.save(output_filename+'.npy', sim_parameters)

    # assign data generation parameters
    seedlist = range(sim_parameters['num_seeds'])
    num_steps = sim_parameters['num_episodes']*episode_length
    num_warmup_steps = 100

    # Initialize Groundmodel
    model = GroundModelJointPolicy(
        num_agents,
        state_space_dim,
        action_space_dim=action_space_dim,
        model_paras=sys_parameters['jointagent_groundmodel_paras']
    )

    # Initialize environment
    dummy_seed = 1
    env = Environment(state_space_dim, num_agents, episode_length,
                      fluctuation_strength_factor=fluctuation_strength_factor, start_seed=dummy_seed)

    # rollout model into a dataset of trajectories
    st = time.time()
    for seed in seedlist:
        print(f"running seed {seed} of {len(seedlist)}")
        episode_time_indices = []
        state_seq = []
        joint_action_seq = []
        env.state = env.sample_initial_state(seed=seed)

        # warmup
        for step in range(num_warmup_steps):
            observed_state = env.state[:state_space_dim]
            action_probability_vectors = torch.squeeze(model.forward(torch.unsqueeze(observed_state,dim=0)),dim=0)
            if action_selection == 'greedy':  # take greedy action
                actions = torch.argmax(action_probability_vectors, dim=-1)
            else:  # sample
                # Categorical has batch functionality!
                actions = Categorical(action_probability_vectors)
            env.state, episode_step = env.step(env.state, actions)

        for step in range(num_steps):

            observed_state = env.state[:state_space_dim]
            state_seq.append(observed_state.detach().cpu().numpy())
            action_probability_vectors = torch.squeeze(model.forward(torch.unsqueeze(observed_state,dim=0)),dim=0)

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
        filename = f'{output_filename}_dataseed_{seed}.npy'
        print('saving '+filename)
        np.save(filename, sim_data)
    print('took '+str(time.time()-st))

if __name__ == '__main__':

    output_path = os.path.join(os.getcwd(), 'output/')

    dataset_label = "4agentdebug"
    groundmodel_name = "bitpop"

    sys_parameters = {}
    sys_parameters['N'] = args.N  # agents
    # 2^K states so 2^{K+1}possible single agent policies. Here, set so 10*N number of policies >> N  # state space
    sys_parameters['K'] = int(5*np.log2(np.log2(sys_parameters['N'])))
    print("setting state space K=5*log2(log2(N))="+str(sys_parameters['K'])+" dimensions (2^K="+str(2**sys_parameters['K'])+' possible observations)')
    # stability transition control parameter
    sys_parameters['fluctuation_strength_factor'] = args.stablefac
    # number of discrete actions; fixed to 2 for simplicity
    sys_parameters['action_space_dim'] = 2

    jointagent_groundmodel_paras = {}
    jointagent_groundmodel_paras['modelname'] = groundmodel_name

    if groundmodel_name == "bitpop":
        jointagent_groundmodel_paras["corr"] = args.corr  # action pair correlation
        jointagent_groundmodel_paras['ensemble'] = 'sum'
        jointagent_groundmodel_paras['M'] = args.M  # number of agent groups
        assert (sys_parameters['N']/jointagent_groundmodel_paras['M']).is_integer(), \
            "number of agents groups should divide total number of agents for some groundmodels"

    else:
        os.abort("select an implemented groundmodel")
    sys_parameters['jointagent_groundmodel_paras'] = jointagent_groundmodel_paras

    sim_parameters = {}
    sim_parameters['actsel'] = 'greedy'
    sim_parameters['episode_length'] = args.T
    sim_parameters['num_episodes'] = 1
    sim_parameters['num_seeds'] = 2

    # add parameter setting of ground model to label
    dataset_label += ''.join(['_'+key+'_'+str(value)
                              for key, value in jointagent_groundmodel_paras.items()])

    generate_system_data(sys_parameters, sim_parameters,
                         output_path, dataset_label)
