from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import yaml
import torch


def load_data(data_hash, data_seed=0):
    data_dir = f"output/{data_hash}"
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


class CustomDataset(Dataset):
    def __init__(self, states, actions, context_window_size, num_actions):
        self.states = [
            torch.tensor(states[i:i+context_window_size+1]).float() 
            for i in range(len(states) - context_window_size)
            ]
        self.states = torch.stack(self.states) # shape: num_seqs, context_window_size, dim_state
        self.actions = [
            torch.tensor(actions[i:i+context_window_size+1]).long() 
            for i in range(len(actions) - context_window_size)
            ]
        self.actions = torch.stack(self.actions) # shape: num_seqs, context_window_size, num_agents
        self.context_window_size = context_window_size
        self.num_actions = num_actions
        self.state_space_dim = states.shape[1]
        self.num_agents = actions.shape[1]
        self.context_size = self.context_window_size * (self.state_space_dim + self.num_actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):

        states = self.states[idx] # seqlen, statedim
        actions = self.actions[idx] # [:, split[idx]] # (seqlen, num_agents)

        current_state = states[-1] # statedim
        past_states = states[:-1].flatten(start_dim=0) # seqlen*statedim

        target_actions = actions[-1].flatten() # (num_agents,)
        past_action_onehots = torch.nn.functional.one_hot(
                torch.transpose(actions[:-1],-1,-2), 
                num_classes=self.num_actions
            ).flatten(start_dim=-2) #(num_agents, seqlen*num_actions)

        agent_contexts = torch.cat([
            torch.unsqueeze(past_states,dim=0).repeat(self.num_agents,1), 
            past_action_onehots.float()
            ],dim=-1) #(num_agents, seqlen*(statedim + num_actions))
        
        return agent_contexts, current_state, target_actions


def get_seqdata_loaders(data, context_window_size, num_actions=2, train2test_ratio=0.8, batch_size=8):
    states = data["states"]
    actions = data["actions"]
    # shape of states: (num_states, dim_teacher_inp)
    # shape of actions: (num_states, num_teachers)
    print(f"state data dims: {states.shape}, action data dims: {actions.shape}")

    splt=int(train2test_ratio*states.shape[0])
    print(f"using {splt}/{states.shape[0]-splt} samples to train/test")
    train_dataset = CustomDataset(states[:splt], actions[:splt], context_window_size, num_actions)
    test_dataset = CustomDataset(states[splt:], actions[splt:], context_window_size, num_actions)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader