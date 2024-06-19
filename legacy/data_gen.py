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
from collections import OrderedDict
from Environment import Environment
from GroundModelJointPolicy import GroundModelJointPolicy
from utils import numpy_scalar_to_python

# parser = argparse.ArgumentParser(description='data generation parameters')
# parser.add_argument('--T', type=int,
#                     default=1000, help='sample size')
# parser.add_argument('--corr', type=float,
#                     default=1.0, help='action correlation')
# parser.add_argument('--A', type=int,
#                     default=2, help='number of actions')
# parser.add_argument('--N', type=int,
#                     default=10, help='number of ground agents')
# parser.add_argument('--M', type=int,
#                     default=2, help='number of abstract agents')
# parser.add_argument('--K', type=int,
#                     default=3, help='state space dimension')
# parser.add_argument('--num_seeds', type=int,
#                     default=10, help='number of seeds')
# parser.add_argument('--action_selection_method', type=str,
#                     default='greedy', help='action selection method')
# parser.add_argument('--ensemble', type=str,
#                     default='sum', help='ensemble method (sum or mix)')
# parser.add_argument('--ground_model_name', type=str,
#                     default='bitpop', help='ground model name or hashes for loaded model')
# parser.add_argument('--output', type=str,
#                     default='output/', help='output directory')
# args = parser.parse_args()


def generate_synthetic_data(args,policy_params):

    # hard-coded over all states in policy function

    N = args['N'] # agents
    K = args['K'] # state space dimension
    A = args['A'] # number of actions
    T = args['T'] # number of samples
    
    datasets = {}
    seed_list = range(args['num_seeds'])
    for ix, seed in enumerate(seed_list):
        print(f"running seed {seed} of {len(seed_list)}")
        rng = np.random.default_rng(seed=seed)
        policy_params['rng']=rng
        # Initialize ground model
        torch.manual_seed(seed)
        model = GroundModelJointPolicy(
            num_agents=N,
            state_space_dim=K,
            action_space_dim=A,
            )
        model.set_action_policies(policy_params)
        
        actions = []
        states= 2*rng.uniform(size=(T,K)).astype(np.float32)-1
        # for state in model.state_set:
        for state in states:
            action_probability_vectors = torch.squeeze(
                model.forward(torch.unsqueeze(torch.Tensor(state),dim=0))
                ,dim=0)
            actions.append(torch.argmax(action_probability_vectors, dim=-1))
        actions = np.array(torch.vstack(actions))

        # save data
        shuffled_inds= rng.permutation(T)
        datasets[f"dataset_{seed}"] = { 
            "seed": seed, 
            "states": states[shuffled_inds], 
            "actions": actions[shuffled_inds],
            "preferred_actions": model.preferred_actions
            }
    return datasets

def generate_data(args):

    output_path = os.path.join(os.getcwd(), args['output'])
    os.makedirs(output_path, exist_ok=True)

    policy_params = {}
    policy_params['ground_model_name'] = args['ground_model_name']
    print(policy_params['ground_model_name'])
    if policy_params['ground_model_name'] == "bitpop":
        policy_params["corr"] = args['corr']  # action pair correlation
        policy_params['ensemble'] = args['ensemble']  # ensemble method
        policy_params['M'] = args['M']  # number of agent groups
        assert (args['N']/policy_params['M']).is_integer(), \
            "number of agents groups should divide total number of agents for some ground models"
    elif '/' in policy_params['ground_model_name']: # ['single', 'match','multi']:
        data_hash,train_hash = args['ground_model_name'].split('/')
        outdir = 'output/'
        data_dir = f'data_{data_hash}/'
        config_filename = os.path.join(outdir, data_dir, 'config.yaml')
        with open(config_filename, 'r') as f:
            data_settings = yaml.load(f, Loader=yaml.FullLoader)['file_attrs']
        print('data_settings:')
        print(data_settings)
        for key,value in data_settings.items():
            args[key]=value
        policy_params['data_settings'] = data_settings

        save_dir = os.path.join(outdir, data_dir, 'training_results/')
        train_info_dir = os.path.join(save_dir, train_hash)
        policy_params["loadedmodel_path"] =train_info_dir
        with open(train_info_dir + "/args.yaml", 'r') as f:
            training_args = yaml.load(f, Loader=yaml.FullLoader)
        print('orig_training_args:')
        print(training_args)
        args['origtraining_args_path'] = train_info_dir + "/args.yaml" #this novel field makes args seed a new hash
        policy_params['origtraining_args'] = training_args
        print(f"parameters overwritten from {training_args['model_name']} model at {args['ground_model_name']}")
    else:
        os.abort("select an implemented ground model")

    # assign sim parameters
    action_selection = args['action_selection_method']

    datasets = generate_synthetic_data(args,policy_params)

    # take all args except output path
    hash_dict = args.copy()
    hash_dict.pop('output')

    # make dash_dict an ordered dict
    hash_dict = dict(sorted(hash_dict.items()))

    # get the hash of the hash_dict
    hash_var = hashlib.blake2s(str(hash_dict).encode(), digest_size=5).hexdigest()
    # get a timestamp - use this to [n] either make the output folder unique or [y] as file metadata
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # combine the hash to get a unique filename
    output_filename = f"data_{hash_var}"
    print('saving '+output_filename)

    output_dir = os.path.join(output_path, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # save the data
    filename = os.path.join(output_dir, "data" + '.h5')
    attrs_filename = os.path.join(output_dir, "config" + '.yaml')

    with h5py.File(filename, 'w') as f, open(attrs_filename, 'w') as yaml_file:
        f.attrs.update(args)
        f.attrs['timestamp'] = timestamp
        attrs_dict = {'file_attrs': {k: numpy_scalar_to_python(v) for k, v in f.attrs.items()}}
        attrs_dict['file_attrs']['hash'] = hash_var
        for dataset_name, dataset in datasets.items():
            group = f.create_group(dataset_name)
            for key, value in dataset.items():
                group.create_dataset(key, data=value)
        yaml.dump(attrs_dict, yaml_file)

    return hash_var

# if __name__ == '__main__':

#     generate_data(vars(args))