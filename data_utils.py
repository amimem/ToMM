from torch.utils.data import Dataset
import h5py
import numpy as np
import yaml
import torch
from models import logit
from types import SimpleNamespace
import os
import hashlib
import time
import numpy as np

def numpy_scalar_to_python(value):
    if isinstance(value, np.generic):
        return value.item()
    return value

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

    train_dataset = datasets[f"train_dataset_{data_seed}"]
    test_dataset = datasets[f"test_dataset_{data_seed}"]
    return train_dataset, test_dataset, config


class ContextDataset(Dataset):
    def __init__(self, data, seq_len, num_actions, check_duplicates=False):

        states = data["states"]
        actions = data["actions"]

       
        """
        num_samples=len(states)
        self.states = np.empty((num_samples, seq_len, states.shape[1]))
        self.actions = np.empty((num_samples, seq_len, actions.shape[1]))

        mask=np.ones(num_samples,dtype=bool)
        index_list = np.arange(num_samples) 
        for state_id in index_list:
            mask[state_id] = False
            indices_for_context=np.random.choice(index_list[mask],size=seq_len-1,replace=False)
            self.states[state_id]=np.vstack([states[indices_for_context],states[state_id]])
            self.actions[state_id]=np.vstack([actions[indices_for_context],actions[state_id]])
            mask[state_id] = True
        self.states = torch.from_numpy(self.states).float()# num_seqs, seq_len, state_dim
        self.actions = torch.from_numpy(self.actions).long()# num_seqs, seq_len, num_agents
        """
        
        self.states = [
            torch.tensor(states[i:i+seq_len]).float()
            for i in range(len(states) - seq_len)]
        self.actions = [
            torch.tensor(actions[i:i+seq_len]).long()
            for i in range(len(actions) - seq_len)]
        self.states = torch.stack(self.states)# num_seqs, seq_len, state_dim
        self.actions = torch.stack(self.actions)# num_seqs, seq_len, num_agents

        if check_duplicates:
            print('Checking duplicates: ')
            has_context_duplicates = torch.zeros(len(self.states))
            tmpacts=self.actions.transpose(1,2)
            for i in range(len(self.actions)): 
                if i % 1000 == 0:
                    print(f"{int((i/len(self.actions))*100)}%",end="\r")
                for ag_i in range(actions.shape[1]):
                    for ag_j in range(actions.shape[1]):
                        if ag_i!=ag_j:
                            if torch.all(tmpacts[i,ag_i]==tmpacts[i,ag_j]):
                                has_context_duplicates[i] = True
                                break
                    else:
                        continue
                    break
            self.number_of_contexts_with_duplicates =int(torch.sum(has_context_duplicates))
            print(f'\n{self.states.shape[:2]} shaped contexts with {self.number_of_contexts_with_duplicates} duplicates')

        self.seq_len = seq_len
        self.num_actions = num_actions
        self.state_dim = states.shape[1]
        self.num_agents = actions.shape[1]

    def check_duplicates(self):
        print('Checking duplicates: ')
        # Initialize tensor to track duplicates
        has_context_duplicates = torch.zeros(len(self.states), dtype=torch.bool)
        # Transpose actions for easier comparison
        tmpacts = self.actions.transpose(1, 2)
        # Iterate over each state
        for i in range(len(self.actions)):
            if i % 1000 == 0:
                print(f"{int((i / len(self.actions)) * 100)}%", end="\r")
            # Use broadcasting to compare each agent's actions with every other agent's actions in a vectorized manner
            comparisons = tmpacts[i].unsqueeze(1) == tmpacts[i].unsqueeze(0)
            # Sum over the action dimension to find complete matches, and then check for any matches across agents
            complete_matches = comparisons.all(dim=2).sum(dim=0) > 1
            # If any complete matches are found, mark this state as having duplicates
            if complete_matches.any():
                has_context_duplicates[i] = True

        self.number_of_contexts_with_duplicates = int(torch.sum(has_context_duplicates))
        print(f'\n{self.states.shape[:2]} shaped contexts with {self.number_of_contexts_with_duplicates} duplicates')
        
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


def gen_logit_dataset(config):

    datasets = {}
    data_seed_list = range(2)
    for ix, data_seed in enumerate(data_seed_list):
        print(f"running seed {data_seed} of {len(data_seed_list)}")
        rng = np.random.default_rng(seed=data_seed)

        model=logit(SimpleNamespace(**config),rng)
        for label in ['train','test']:
            states=sample_states(config[f'num_{label}_samples'],config['state_dim'],rng)
            actions= []
            for state in states:
                action_probability_vectors = model.forward(state)
                actions.append(np.argmax(action_probability_vectors, axis=-1))
            actions = np.vstack(actions)
            # save data
            shuffled_inds= rng.permutation(config[f'num_{label}_samples'])
            datasets[f"{label}_dataset_{data_seed}"] = { 
                "data_seed": data_seed, 
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

    # def load_model(model_config, train_dir):
    #     model = STOMP(model_config).to(device)
    #     model.load_state_dict(torch.load(train_dir + "/state_dict_final.pt"))
    #     return model