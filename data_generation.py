import torch
import numpy as np
import os
import argparse
import time
import itertools
import h5py
import hashlib
import yaml
from torch.distributions import Categorical
from Environment import Environment
from GroundModelJointPolicy import GroundModelJointPolicy
from utils import numpy_scalar_to_python

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

    N = args.N  # agents
    K = args.K  # state space dimension
    A = args.A # number of actions
    args.T = args.sps*(A**K) # overwritten when not using environment

    datasets = {}
    seed_list = range(args.num_seeds)
    for ix, seed in enumerate(seed_list):
        print(f"running seed {seed} of {len(seed_list)}")
        rng = np.random.default_rng(seed=seed)
        policy_params['seed'] = seed

        # Initialize ground model
        torch.manual_seed(seed)
        model = GroundModelJointPolicy(
            num_agents=N,
            state_space_dim=K,
            action_space_dim=A,
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

        # save data
        shuffled_inds= rng.permutation(args.T)
        datasets[f"dataset_{seed}"] = { "seed": seed, 
                                        "states": state_seq[shuffled_inds], 
                                        "actions": joint_action_seq[shuffled_inds],
                                        "timesteps": episode_time_indices[shuffled_inds] }

    return datasets


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
    datasets = {}
    for ix, seed in enumerate(seed_list):
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

        # save data
        datasets[f"dataset_{seed}"] = { "seed": seed,
                                        "states": np.array(state_seq),
                                        "actions": np.array(joint_action_seq),
                                        "timesteps": np.array(episode_time_indices) }
        
    print('took '+str(time.time()-st))
    return datasets

if __name__ == '__main__':

    output_path = os.path.join(os.getcwd(), args.output)
    os.makedirs(output_path, exist_ok=True)

    policy_params = {}
    policy_params['model_name'] = args.ground_model_name
    if policy_params['model_name'] == "bitpop":
        policy_params["corr"] = args.corr  # action pair correlation
        policy_params['ensemble'] = args.ensemble  # ensemble method
        policy_params['M'] = args.M  # number of agent groups
        assert (args.N/policy_params['M']).is_integer(), \
            "number of agents groups should divide total number of agents for some ground models"
    else:
        os.abort("select an implemented ground model")

    # assign sim parameters
    num_episodes = args.num_episodes
    action_selection = args.action_selection_method

    # 2^K states so 2^{K+1}possible single agent policies. Here, set so 10*N number of policies >> N  # state space
    # K_bound = int(5*np.log2(np.log2(args.N)))  # bound on state space dimension
    print("setting state space K=5*log2(log2(N))="+str(args.K)+" dimensions (2^K="+str(2**args.K)+' possible observations)')

    if args.env:
        config = {"num_episodes": num_episodes, "action_selection": action_selection}
        datasets = generate_simulated_data(policy_params, config)
    else:
        datasets = generate_synthetic_data(policy_params)

    # get the hash of the arguments
    hash = hashlib.blake2s(str(args.__dict__).encode(), digest_size=5).hexdigest()
    # get a timestamp - use this to [n] either make the output folder unique or [y] as file metadata
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # combine the hash to get a unique filename
    output_filename = f"data_{hash}"
    print('saving '+output_filename)

    output_dir = os.path.join(output_path, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # save the data
    filename = os.path.join(output_dir, "data" + '.h5')
    attrs_filename = os.path.join(output_dir, "config" + '.yaml')

    with h5py.File(filename, 'w') as f, open(attrs_filename, 'w') as yaml_file:
        f.attrs.update(args.__dict__)
        f.attrs['timestamp'] = timestamp
        attrs_dict = {'file_attrs': {k: numpy_scalar_to_python(v) for k, v in f.attrs.items()}}
        for dataset_name, dataset in datasets.items():
            group = f.create_group(dataset_name)
            for key, value in dataset.items():
                group.create_dataset(key, data=value)
        yaml.dump(attrs_dict, yaml_file)