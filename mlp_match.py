import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import h5py
import yaml
import torch
import wandb
import time
import os
from models import MLP,MatchNet

# get slurm job array index
try:
    job_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
except:
    job_id = -1
    print("Not running on a cluster")

seed = 0

# sets the seed for generating random numbers
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# create a Dataset object
class CustomDataset(Dataset):
    def __init__(self, states, actions, context_window_size, num_actions, seed=0):
        # get a rng
        self.rng = np.random.RandomState(seed)
        self.states = [torch.tensor(states[i:i+context_window_size+1]).float() for i in range(len(states) - context_window_size)]
        self.states = torch.stack(self.states) # shape: num_seqs, context_window_size, dim_state
        self.actions = [torch.tensor(actions[i:i+context_window_size+1]).long() for i in range(len(actions) - context_window_size)]
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

        # print(torch.unsqueeze(past_states,dim=0).repeat(self.num_agents,1).shape)
        # print(past_action_onehots.float().shape)
        agent_contexts = torch.cat([
            torch.unsqueeze(past_states,dim=0).repeat(self.num_agents,1), 
            past_action_onehots.float()
            ],dim=-1) #(num_agents, seqlen*(statedim + num_actions))
        
        # print(agent_contexts.shape)
        return agent_contexts, current_state, target_actions
    
def eval_perf(model, dataloader, and_train=False):
    if and_train:
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    else:
        model.eval()

    criterion = nn.CrossEntropyLoss()
    batch_loss = []
    batch_accuracy = []
    for i, (agent_contexts, current_state, target_actions) in enumerate(dataloader):
        # agent_contexts: (bsz, num_agents, context_window_size*(state_space_dim+num_actions))
        # current_state: (bsz, state_space_dim)
        # targets: (bsz, num_agents)
    
        if model.name=='match':
            #no task context, only training memory
            if i==0: #waste first batch to initialize training memory.
                state_history=current_state.detach().clone()
                joint_action_history=torch.nn.functional.one_hot(target_actions, num_classes=dataloader.dataset.num_actions).float()
                continue
            else:
                action_logit_vectors = model(current_state,state_history,joint_action_history)
                state_history=torch.cat((state_history,current_state))
                joint_action_history=torch.cat((joint_action_history,torch.nn.functional.one_hot(target_actions, num_classes=dataloader.dataset.num_actions).float()))         
        elif model.name=='mlp':
            #no training memory, full per-agent task context
            # print( torch.unsqueeze(current_state,
                   ` # dim=1).repeat((1,dataloader.dataset.num_agents,1)).shape)
            # print(agent_contexts.shape)
            parallel_inputs = torch.cat([
                torch.unsqueeze(current_state,
                    dim=1).repeat((1,dataloader.dataset.num_agents,1)), #repeat state N times
                agent_contexts,
            ], dim=-1)
            action_logit_vectors = model(parallel_inputs)
        else:
            abort("not a valid model name")
        
        if and_train:
            optimizer.zero_grad()
        
        loss = sum(
            criterion(action_logit_vectors[:,agent_idx,:], target_actions[:,agent_idx])
            for agent_idx in range(dataloader.dataset.num_agents)
            )

        if and_train:
            loss.backward()
            optimizer.step()

        accuracy = (torch.argmax(action_logit_vectors, dim=-1) == target_actions).float().mean().item()
        batch_loss.append(loss.item())
        batch_accuracy.append(accuracy)
        
    return np.mean(batch_loss), np.mean(batch_accuracy)


def get_seqdata_loaders(data, context_window_size, num_actions=2, train2test_ratio=0.8, batch_size=8):
    states = data["states"]
    actions = data["actions"]
    # shape of states: (num_states, dim_teacher_inp)
    # shape of actions: (num_states, num_teachers)
    print(f"state data dims: {states.shape}, action data dims:{actions.shape}")

    splt=int(train2test_ratio*states.shape[0])
    print(f"using {splt}/{states.shape[0]-splt} samples to train/test")
    train_dataset = CustomDataset(states[:splt], actions[:splt], context_window_size, num_actions, seed=seed)
    test_dataset = CustomDataset(states[splt:], actions[splt:], context_window_size, num_actions, seed=seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

def init_model(model_config):

    # Create an instance of the MLP
    context_size = model_config['context_size']
    state_space_dim = model_config['state_space_dim']
    num_actions = model_config['num_actions']
    num_agents = model_config['num_agents']
    hidden_size = model_config['hidden_size']
    num_hidden_layers = model_config['num_hidden_layers']

    if model_config['model_name']=='mlp':
        input_size=context_size+state_space_dim
        model = MLP(input_size, hidden_size, num_actions, num_hidden_layers)
    elif model_config['model_name']=='match':
        num_groups = model_config['M']
        model = MatchNet(
            state_space_dim=state_space_dim,
            hidden_size=hidden_size,
            num_ground_agents=num_agents,
            num_abstract_agents=num_groups,
            num_layers=num_hidden_layers,
        )     
    else:
        abort("Choose valid model name")

    return model

if __name__ == "__main__":
    set_seed(seed)

    # for n_agents and n_groups = n_agents
    # data_hashes = ['data_75816d80b9']
    data_hashes = ['data_a3ea2acb24']
    model_config = {}
    # model_config['model_name'] = 'match'
    model_config['model_name'] = 'mlp'
    model_config['M'] = 1

    # make sure all data_hashes are in the output folder
    assert all([os.path.exists(f"output/{data_hash}") for data_hash in data_hashes])
    print("All data hashes are in the output folder")

    context_window_sizes = [8]
    w_d = [(256, 2)]
    num_epochs = 200

    df = pd.DataFrame(columns=[
    "data_hash", 
    "num_agents", 
    "num_groups", 
    "state_space_dim", 
    "num_actions",
    "model_name", 
    "context_window_size", 
    "hidden_size", 
    "num_hidden_layers", 
    "epoch", 
    "loss", 
    "accuracy"
    ])

    # context_window_sizes = [context_window_sizes[job_id]]
    # print(f"Running context length {context_window_sizes}")

    if job_id != -1:
        data_hashes = [data_hashes[job_id]]

    for data_hash in data_hashes:
        for context_window_size in context_window_sizes:
            for hidden_size, num_hidden_layers in w_d:
                model_config['hidden_size'] = hidden_size
                model_config['num_hidden_layers'] = num_hidden_layers
                data, config = load_data(data_hash)

                config.update({"context_window_size": context_window_size, "hidden_size": hidden_size, "num_hidden_layers": num_hidden_layers})
                wandb.init(project="MLP", group="testmatch", job_type=None, config=config)
                
                train_dataloader, test_dataloader = get_seqdata_loaders(data, context_window_size)
                model_config['context_size'] = train_dataloader.dataset.context_size
                model_config['num_actions'] = train_dataloader.dataset.num_actions
                model_config['num_agents'] = train_dataloader.dataset.num_agents
                model_config['state_space_dim'] = train_dataloader.dataset.state_space_dim
                model = init_model(model_config)

                for epoch in range(num_epochs):

                    train_epoch_loss, train_epoch_accuracy = eval_perf(model, train_dataloader, and_train=True)
                    test_epoch_loss, test_epoch_accuracy = eval_perf(model, test_dataloader)
                    
                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.4f},\
                          Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_accuracy:.4f}")
                    wandb.log({"epoch": epoch, "train_loss": train_epoch_loss, "train_accuracy": train_epoch_accuracy, \
                                "test_loss": test_epoch_loss, "test_accuracy": test_epoch_accuracy})
                    
                    # Append the data to the DataFrame
                    new_row = pd.DataFrame({
                        "data_hash": [data_hash],
                        "num_agents": [config["file_attrs"]["N"]],#["num_teachers"]],
                        "num_groups": [config["file_attrs"]["M"]],#["num_groups"]],
                        "state_space_dim": [config["file_attrs"]["K"]],#["dim_state"]],
                        "num_actions": [config["file_attrs"]["A"]],
                        "model_name": [model_config['model_name']],
                        "context_window_size": [context_window_size],
                        "hidden_size": [model_config['hidden_size']],
                        "num_hidden_layers": [model_config['num_hidden_layers']],
                        "epoch": [epoch],
                        "train_loss": [train_epoch_loss],
                        "train_accuracy": [train_epoch_accuracy],
                        "test_loss": [test_epoch_loss],
                        "test_accuracy": [test_epoch_accuracy]
                    }, index=[0])
                    df = pd.concat([df, new_row], ignore_index=True)
                wandb.finish()

    time_str = time.strftime("%Y-%m-%d-%H-%M")
    df.to_csv(f"output/{time_str}_results.csv", index=False)