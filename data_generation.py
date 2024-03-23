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
import itertools
import random

parser = argparse.ArgumentParser(description='data generation parameters')
parser.add_argument('--stablefac', type=float,
                    default=8.0, help='stability factor')
parser.add_argument('--sps', type=int,
                    default=16, help='samples per state')
parser.add_argument('--T', type=int,
                    default=1000, help='episode length')
parser.add_argument('--corr', type=float,
                    default=1.0, help='action correlation')
parser.add_argument('--A', type=int,
                    default=2, help='number of actions')
parser.add_argument('--N', type=int,
                    default=4, help='number of ground agents')
parser.add_argument('--M', type=int,
                    default=2, help='number of abstract agents')
parser.add_argument('--K', type=int,
                    default=3, help='state space dimension')
parser.add_argument('--num_episodes', type=int,
                    default=1, help='number of episodes')
parser.add_argument('--num_seeds', type=int,
                    default=10, help='number of seeds')
parser.add_argument('--action_selection_method', type=str,
                    default='greedy', help='action selection method')
parser.add_argument('--ensemble', type=str,
                    default='sum', help='ensemble method (sum or mix)')
parser.add_argument('--ground_model_name', type=str,
                    default='bitpop', help='ground model name')
parser.add_argument('--env', type=bool,
                    default=False, help='use environment')
parser.add_argument('--output', type=str,
                    default='output/', help='output directory')
args = parser.parse_args()


def generate_synthetic_data(policy_params):

    # hard-coded over all states in policy function

    K = args.K  # state space dimension
    A = args.A # number of actions
    args.T = args.sps*(A**K) # overwritten when not using environment

    data_list = []

    seed_list = range(args.num_seeds)
    for seed in seed_list:
        print(f"running seed {seed} of {len(seed_list)}")

        rng = np.random.default_rng(seed=seed)
        policy_params['seed'] = seed

        # Initialize ground model
        torch.manual_seed(seed)
        model = GroundModelJointPolicy(
            num_agents,
            state_space_dim,
            action_space_dim=action_space_dim,
            )
        model.set_action_policies(policy_params)
        
        states = np.array([np.array(l) for l in list(map(list, itertools.product(
            range(A), repeat=K)))]).astype(np.single)
        actions = []
        for state in model.state_set:
            action_probability_vectors = torch.squeeze(model.forward(torch.unsqueeze(torch.Tensor(state),dim=0)),dim=0)
            actions.append(torch.argmax(action_probability_vectors, dim=-1))
        actions = np.array(torch.vstack(actions))

        state_seq = np.tile(states, reps=(args.sps, 1))
        joint_action_seq = np.tile(actions, reps=(args.sps, 1))
        episode_time_indices = np.arange(args.T)
        sim_data = {}
        sim_data["seed"] = seed
        sim_data["times"] = episode_time_indices
        shuffled_inds= rng.permutation(args.T)
        sim_data["states"] = state_seq[shuffled_inds]
        sim_data["actions"] = joint_action_seq[shuffled_inds]
        data_list.append(sim_data)

    return data_list


def generate_simulated_data(policy_params, config, warmup=False):
    
    # assign system parameters
    state_space_dim = args.K   # state space dimension
    num_agents = args.N   # agents
    fluctuation_strength_factor = args.stablefac
    action_space_dim = args.A # actions

    # assign sim parameters
    episode_length = args.T
    num_episodes = config['num_episodes']
    action_selection = config['action_selection']

    # assign data generation parameters
    seed_list = range(args.num_seeds)
    num_steps = num_episodes*episode_length
    num_warmup_steps = 100

    dummy_seed = 1
    policy_params['seed'] = dummy_seed

    # set seeds for reproducibility
    torch.manual_seed(dummy_seed)

    # Initialize Groundmodel
    model = GroundModelJointPolicy(
        num_agents,
        state_space_dim,
        action_space_dim=action_space_dim,
    )
    model.set_action_policies(policy_params)

    # Initialize environment
    env = Environment(state_space_dim, num_agents, episode_length,
                      fluctuation_strength_factor=fluctuation_strength_factor, start_seed=dummy_seed)

    # rollout model into a dataset of trajectories
    st = time.time()
    data_list = []
    for seed in seed_list:
        print(f"running seed {seed} of {len(seed_list)}")
        if not warmup:
            rng = np.random.default_rng(seed=seed)
            M = policy_params['M']
            K = args.K
            A = args.A
            states = np.array([np.array(l) for l in list(map(list, itertools.product(
                range(A), repeat=K)))]).astype(np.single)
            actions = rng.integers(0, A, [len(states), M]).astype(int)

            actions = np.vstack(
                # (actions[:, 0], actions[:, 0], actions[:, 1], actions[:, 1])).T
                (actions[:, 0], (~actions[:, 0].astype(bool)).astype(int), actions[:, 1], (~actions[:, 1].astype(bool)).astype(int))).T
            samples_per_state = args.sps # batch_size
            state_seq = np.tile(states, reps=(samples_per_state, 1))
            joint_action_seq = np.tile(actions, reps=(samples_per_state, 1))
            episode_time_indices = np.arange(samples_per_state*A**K)
        else:
            episode_time_indices = []
            state_seq = []
            joint_action_seq = []
            env.state = env.sample_initial_state(seed=seed)

            # warmup
            for _ in range(num_warmup_steps):
                observed_state = env.state[:state_space_dim]
                action_probability_vectors = torch.squeeze(model.forward(torch.unsqueeze(observed_state,dim=0)),dim=0)
                if action_selection == 'greedy':  # take greedy action
                    actions = torch.argmax(action_probability_vectors, dim=-1)
                else:  # sample
                    # Categorical has batch functionality!
                    actions = Categorical(action_probability_vectors)
                env.state, episode_step = env.step(env.state, actions)

            for _ in range(num_steps):
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
        sim_data["seed"] = seed
        sim_data["times"] = np.array(episode_time_indices)
        sim_data["states"] = np.array(state_seq)
        sim_data["actions"] = np.array(joint_action_seq)

        data_list.append(sim_data)
    print('took '+str(time.time()-st))
    return data_list

if __name__ == '__main__':

    output_path = os.path.join(os.getcwd(), args.output)

    # remove
    dataset_label = "4agentdebug"
    # 2^K states so 2^{K+1}possible single agent policies. Here, set so 10*N number of policies >> N  # state space
    # K_bound = int(5*np.log2(np.log2(args.N)))
    print("setting state space K=5*log2(log2(N))="+str(args.K)+" dimensions (2^K="+str(2**args.K)+' possible observations)')

    policy_params = {}
    policy_params['model_name'] = args.ground_model_name
    if policy_params['model_name'] == "bitpop":
        policy_params["corr"] = args.corr  # action pair correlation
        policy_params['ensemble'] = args.ensemble  # ensemble method
        policy_params['M'] = args.M  # number of agent groups
        assert (args.N/policy_params['M']).is_integer(), \
            "number of agents groups should divide total number of agents for some groundmodels"
    else:
        os.abort("select an implemented ground model")
    # add parameter setting of ground model to label
    dataset_label += ''.join(['_'+key+'_'+str(value)
                              for key, value in policy_params.items()])

    # assign sim parameters
    num_episodes = args.num_episodes
    action_selection = args.action_selection_method
    
    # assign system parameters
    state_space_dim = args.K   # state space dimension
    num_agents = args.N   # agents
    action_space_dim = args.A # actions

    if args.env:
        config = {"num_episodes": num_episodes, "action_selection": action_selection}
        data_list = generate_simulated_data(policy_params, config)
        output_filename = f"{output_path}_{dataset_label}_sim_data_action_selection_{action_selection}_numepi_{num_episodes}_K_{state_space_dim}_N_{num_agents}_T_{args.T}_g_{args.stablefac}]"
    else:
        data_list = generate_synthetic_data(policy_params)
        output_filename = f"{output_path}_{dataset_label}_sim_data_action_selection_{action_selection}_numepi_{num_episodes}_K_{state_space_dim}_N_{num_agents}_T_{args.T}_sps_{args.sps}"

    for sit,sim_data in enumerate(data_list):
        filename = f'{output_filename}_dataseed_{sim_data["seed"]}.npy'
        print('saving '+filename)
        np.save(filename, sim_data)