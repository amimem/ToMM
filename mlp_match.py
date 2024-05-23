import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import wandb
import time
import os
from models import init_model
from data_utils import load_data,get_seqdata_loaders
from utils import get_width

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

#iterates over dataset, evaluating model response and optionally updating parameters
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
        # agent_contexts: (bsz, num_agents, context_size), context_size = context_window_size*(state_space_dim+num_actions)
        # current_state: (bsz, state_space_dim)
        # targets: (bsz, num_agents)
    
        if model.name=='match':
            #no task context, only training memory
            if i==0: #waste first batch to initialize training memory.
                state_history=current_state.detach().clone()
                joint_action_history=torch.nn.functional.one_hot(
                    target_actions, num_classes=dataloader.dataset.num_actions).float()
                continue
            else:
                action_logit_vectors = model(current_state,state_history,joint_action_history)
                state_history=torch.cat((state_history,current_state))
                joint_action_history=torch.cat([
                    joint_action_history,
                    torch.nn.functional.one_hot(target_actions, num_classes=dataloader.dataset.num_actions).float()
                    ])         
        elif model.name=='mlp':
            #no training memory, full per-agent task context
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

def train(data_hash, modeltrain_config):
    
    data, data_config = load_data(data_hash)

    # config.update({
    #     "context_window_size": context_window_size, 
    #     "hidden_size": hidden_size, 
    #     "num_hidden_layers": num_hidden_layers}
    #     )
    modeltrain_config['data_config'] = config
    wandb.init(project="MLP", group="testmatch", job_type='no_encoder', config=config)
    
    train_dataloader, test_dataloader = get_seqdata_loaders(data, model_config['context_window_size'])
    modeltrain_config['context_size'] = train_dataloader.dataset.context_size
    modeltrain_config['num_actions'] = train_dataloader.dataset.num_actions
    modeltrain_config['num_agents'] = train_dataloader.dataset.num_agents
    modeltrain_config['state_space_dim'] = train_dataloader.dataset.state_space_dim

    solver_dict = {"model_name": modeltrain_config['model_name'],
        "n_hidden_layers": modeltrain_config['num_hidden_layers'],
        "num_abs_agents": modeltrain_config['M_train'],
        "abs_action_space_dim": None,
        "n_features": None,
        "num_agents": modeltrain_config['num_agents'],
        "state_space_dim": modeltrain_config['state_space_dim'],
        "action_space_dim": modeltrain_config['num_actions'],
        "enc2dec_ratio": 1,
        "num_parameters": modeltrain_config['P'],
         }
    modeltrain_config['hidden_size'] = int(get_width(solver_dict))
    print(f"hidden_dim={modeltrain_config['hidden_size']}")
    # context_window_sizes = [context_window_sizes[job_id]]
    # print(f"Running context length {context_window_sizes}")
    # if job_id != -1:
    #     data_hashes = [data_hashes[job_id]]

    model = init_model(modeltrain_config)

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

    for epoch in range(model_config['num_epochs']):

        train_epoch_loss, train_epoch_accuracy = eval_perf(model, train_dataloader, and_train=True)
        test_epoch_loss, test_epoch_accuracy = eval_perf(model, test_dataloader)
        
        print(
            f"Epoch {epoch+1}/{model_config['num_epochs']}, "+\
            f"Train Loss: {train_epoch_loss:.4f}, "+\
            f"Train Accuracy: {train_epoch_accuracy:.4f}, "+\
            f"Test Loss: {test_epoch_loss:.4f}, "+\
            f"Test Accuracy: {test_epoch_accuracy:.4f}"
            )
        wandb.log({
            "epoch": epoch, 
            "train_loss": train_epoch_loss, 
            "train_accuracy": train_epoch_accuracy,
            "test_loss": test_epoch_loss, 
            "test_accuracy": test_epoch_accuracy
            })
        
        # Append the data to the DataFrame
        new_row = pd.DataFrame({
            "data_hash": [data_hash],
            "num_agents": [config["file_attrs"]["N"]],
            "num_groups": [config["file_attrs"]["M"]],
            "state_space_dim": [config["file_attrs"]["K"]],
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
    data_dir = f"output/{data_hash}"
    data_filename = f"{data_dir}"
    store_name = f"output/{time_str}_results.csv"
    df.to_csv(store_name, index=False)
    
    return store_name

if __name__ == "__main__":
    
    set_seed(seed)

    hashdata_filename = f"hashlist_K_{K}_Msys_{M_sys}_T_{T}_Mtrain_{M_train}_Ep_{epochs}_dataseed_{data_seed}.npy"
    hash_data=np.load(hashlist_filename)
    df=hash_data['hashes']

    N=10
    K=8
    P=1e6
    corr=1.

    data_hash = df.loc[(df.corr==corr) & (df.N==N) & (hashtype=='simple_data') & (df.P==P),'hash']
    # make sure all data_hashes are in the output folder
    assert all([os.path.exists(f"output/{data_hash}") for data_hash in list(df.hash.values)])
    print("All data hashes are in the output folder")

    modeltrain_config = {}
    modeltrain_config['M_train'] = 1
    modeltrain_config['num_hidden_layers'] = 2
    modeltrain_config['context_window_size'] = 8
    modeltrain_config['num_epochs'] = 200
    modeltrain_config['P'] = 1e6

    modeltrain_config['model_name'] = 'match'
    # modeltrain_config['model_name'] = 'mlp'

    store_name=train(data_hash, model_config)

