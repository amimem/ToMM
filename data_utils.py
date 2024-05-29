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
    def __init__(self, states, actions, seq_len, num_actions):
        self.states = [
            torch.tensor(states[i:i+seq_len]).float()
            for i in range(len(states) - seq_len+1)
        ]
        self.states = torch.stack(self.states)  # num_seqs, seq_len, state_dim
        self.actions = [
            torch.tensor(actions[i:i+seq_len]).long()
            for i in range(len(actions) - seq_len+1)
        ]
        # num_seqs, seq_len, num_agents
        self.actions = torch.stack(self.actions)

        self.seq_len = seq_len
        self.num_actions = num_actions
        self.state_dim = states.shape[1]
        self.num_agents = actions.shape[1]
        self.context_size = self.seq_len * \
            (self.state_dim + self.num_actions) + self.state_dim

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):

        states = self.states[idx]  # seq_len, state_dim
        actions = self.actions[idx]  # seq_len, num_agents

        # current_state = states[-1]  # state_dim
        # state_seq = states[:-1]  # seq_len,state_dim  

        target_actions = actions[-1].flatten()  # num_agents

        action_onehots_seq = torch.nn.functional.one_hot(
            actions,
            num_classes=self.num_actions
        )  # seq_len, num_agents, num_actions
        action_onehots_seq[-1] = 0
        
        # state_action_seqs = torch.cat([
        #     torch.unsqueeze(past_states, dim=0).repeat(
        #         self.num_agents, 1, 1),  # num_agents, seq_len, state_dim
        #     past_action_onehots.float(),  # num_agents, seq_len, num_actions
        #     current_state.repeat(self.num_agents, 1)  # num_agents, state_dim
        # ], dim=-1)  # num_agents, seq_len*(state_dim + num_actions) + state_dim

        return states, action_onehots_seq, target_actions


def get_seqdata_loaders(data, seq_len, num_actions=2, train2test_ratio=0.8, batch_size=8):
    states = data["states"]
    actions = data["actions"]
    # shape of states: (num_states, dim_teacher_inp)
    # shape of actions: (num_states, num_teachers)
    print(f"state data dims: {states.shape}, action data dims: {actions.shape}")

    splt = int(train2test_ratio*states.shape[0])
    print(f"using {splt}/{states.shape[0]-splt} samples to train/test")
    train_dataset = CustomDataset(
        states[:splt], actions[:splt], seq_len, num_actions)
    test_dataset = CustomDataset(
        states[splt:], actions[splt:], seq_len, num_actions)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader
