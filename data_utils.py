from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import yaml
import torch
from models import logit
from types import SimpleNamespace
import os
import hashlib
import time
from utils import numpy_scalar_to_python

def load_data(data_dir, data_seed=0):
    data_filename = f"{data_dir}/data.h5"
    config_filename = f"{data_dir}/config.yaml"
    # load the hdf data
    with h5py.File(data_filename, 'r') as f:
        datasets = {}
        for group_name, group in f.items():
            datasets[group_name] = {key: np.array(
                value) for key, value in group.items()}

    # load the config file
    with open(config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config, flush=True)

    dataset = datasets[f"dataset_{data_seed}"]
    return dataset, config


class CustomSeqDataset(Dataset):
    def __init__(self, states, actions, seq_len, num_actions):

        self.states = [
            torch.tensor(states[i:i+seq_len]).float()
            for i in range(len(states) - seq_len)
        ]
        self.states = torch.stack(self.states)  
        # num_seqs, seq_len, state_dim
        self.actions = [
            torch.tensor(actions[i:i+seq_len]).long()
            for i in range(len(actions) - seq_len)
        ]
        self.actions = torch.stack(self.actions)
        # num_seqs, seq_len, num_agents
        print(f'made {len(self.states)} contexts')

        self.seq_len = seq_len
        self.num_actions = num_actions
        self.state_dim = states.shape[1]
        self.num_agents = actions.shape[1]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        states = self.states[idx]  # seq_len, state_dim
        actions = self.actions[idx]  # seq_len, num_agents 

        target_actions = actions[-1].flatten()  # num_agents

        action_onehots_seq = torch.nn.functional.one_hot(
            actions,
            num_classes=self.num_actions
        )  # seq_len, num_agents, num_actions
        action_onehots_seq[-1] = 0
        
        return states, action_onehots_seq, target_actions


def get_seqdata_loaders(data, seq_len, num_actions=2, train2test_ratio=0.8, batch_size=8):
    states = data["states"]
    actions = data["actions"]
    # shape of states: (num_states, state_dim)
    # shape of actions: (num_states, num_agents)
    print(f"state data dims: {states.shape}, action data dims: {actions.shape}")

    splt = int(train2test_ratio*states.shape[0])
    print(f"using {splt}/{states.shape[0]-splt} samples to train/test")
    train_dataset = CustomSeqDataset(
        states[:splt], actions[:splt], seq_len, num_actions)
    test_dataset = CustomSeqDataset(
        states[splt:], actions[splt:], seq_len, num_actions)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def generate_dataset_from_logitmodel(config):

    datasets = {}
    seed_list = range(2)
    for ix, seed in enumerate(seed_list):
        print(f"running seed {seed} of {len(seed_list)}")
        rng = np.random.default_rng(seed=seed)

        states=sample_states(config['num_samples'],config['state_dim'],rng)
        model=logit(SimpleNamespace(**config),rng)
        actions= []
        for state in states:
            action_probability_vectors = model.forward(state)
            # print(action_probability_vectors.shape)
            actions.append(np.argmax(action_probability_vectors, axis=-1))
        actions = np.vstack(actions)
        # save data
        shuffled_inds= rng.permutation(config['num_samples'])
        datasets[f"dataset_{seed}"] = { 
            "seed": seed, 
            "states": states[shuffled_inds], 
            "actions": actions[shuffled_inds],
            "preferred_actions": model.action_at_corr1
            }

    data_hash=save_dataset_from_model(config, datasets)

    return data_hash

def sample_states(num_samples,state_dim,rng):
    states= 2*rng.uniform(size=(num_samples,state_dim)).astype(np.float32)-1
    return states

def save_dataset_from_model(config, datasets):

    output_path = os.path.join(os.getcwd(), config['outdir'])
    os.makedirs(output_path, exist_ok=True)

    # take all args except output path
    hash_dict = config.copy()
    hash_dict.pop('outdir')

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
        f.attrs.update(config)
        f.attrs['timestamp'] = timestamp
        attrs_dict = {'file_attrs': {k: numpy_scalar_to_python(v) for k, v in f.attrs.items()}}
        attrs_dict['file_attrs']['hash'] = hash_var
        for dataset_name, dataset in datasets.items():
            group = f.create_group(dataset_name)
            for key, value in dataset.items():
                group.create_dataset(key, data=value)
        yaml.dump(attrs_dict, yaml_file)

    return output_filename