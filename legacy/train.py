from torch.utils.data import Dataset, DataLoader
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from utils import MultiChannelNet, count_parameters, get_width
import warnings
import os
import h5py
import yaml
import hashlib
import wandb
import time
# warnings.filterwarnings("ignore", category=FutureWarning)

# parser = argparse.ArgumentParser(description='Training parameters')
# parser.add_argument('--model_name', type=str,
#                     default='stomp', help='Name of the model')
#                     # default='single', help='Name of the model')
#                     # default='multi', help='Name of the model')
# parser.add_argument('--P', type=float, default=1e7,
#                     help='Number of model parameters')
# parser.add_argument('--M', type=int, default=1,
#                     help='Number of abstract agents')
# parser.add_argument('--L', type=int, default=100,
#                     help='Abstract action space dimension')
# parser.add_argument('--n_hidden_layers', type=int, default=2,
#                     help='Number of hidden layers')
# parser.add_argument('--n_features', type=int, default=2,
#                     help='Agent embedding dimension')
# parser.add_argument('--num_codebooks', type=int,
#                     default=10, help='Number of codebooks')
# parser.add_argument('--enc2dec_ratio', type=float,
#                     default=1., help='Encoder to decoder ratio')
# parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
# parser.add_argument('--learning_rate', type=float,
#                     default=5e-5, help='Learning rate')
# parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
# parser.add_argument('--outdir', type=str, default='output/',
#                     help='Output directory')
# parser.add_argument('--data_dir', type=str,
#                     default='data_846a70d602/', help='Data directory')
# parser.add_argument('--seed', type=int, default=0, help='Random seed')
# parser.add_argument('--data_seed', type=int,
#                     default=0, help='data realization')
# parser.add_argument('--use_lr_scheduler', type=bool,
#                     default=False, help='Use learning rate scheduler'
#                     )
# parser.add_argument('--step_LR', type=int,
#                      default=30, help='Step size for learning rate scheduler')
# parser.add_argument('--gamma', type=float,
#                     default=0.1, help='Gamma for learning rate scheduler')
# parser.add_argument('--checkpoint_interval', type=int, default=100, help='Checkpointing interval')
# parser.add_argument('--wandb_entity_name', type=str, default=None, help='sharing of wandb logs')
# parser.add_argument('--wandb_group_name', type=str, default=None, help='group of wandb logs')
# parser.add_argument('--wandb_job_type_name', type=str, default=None, help='job type of wandb logs')

# args = parser.parse_args()

# logging to wandb
# wandb.login()

def train(args):


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device", flush=True)
    seed: int = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    class CustomDataset(Dataset):
        def __init__(self, states, actions):
            self.states = states
            self.actions = actions

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return self.states[idx], self.actions[idx]

    # training setting
    epochs = args['epochs']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    log_interval = args['checkpoint_interval']

    print(f"seed {seed} training of {args['model_name']} model with modelsize {args['P']} for {epochs} epochs using batchsize {batch_size} and LR {args['learning_rate']}", flush=True)

    # load the data from the output folder
    outdir = args['outdir']
    data_dir = args['data_dir']
    data_filename = os.path.join(outdir, data_dir, 'data.h5')
    config_filename = os.path.join(outdir, data_dir, 'config.yaml')
    
    # make a directory for saving the results
    save_dir = os.path.join(outdir, data_dir, 'training_results/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # hash the arguments except for the outdir
    hash_dict = args.copy()
    hash_dict.pop('outdir')

    # make a subdirectory for each run
    hash_var = hashlib.blake2s(str(hash_dict).encode(), digest_size=5).hexdigest()
    train_info_dir = os.path.join(save_dir, hash_var)
    if not os.path.exists(train_info_dir):
        os.makedirs(train_info_dir)

    # initialize wandb
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    wandb_run_name = str(hash_var) + "_" + str(timestamp)
    print("wandb run name: " + wandb_run_name, flush=True)
  
    # share run logs with group using entity label: 'abstraction'
    # wandb.init(
    #     project="STOMP", 
    #     entity=args['wandb_entity_name'], 
    #     group=args['wandb_group_name'], 
    #     job_type=args['wandb_job_type_name'], 
    #     name=wandb_run_name, 
    #     config=args
    #     )

    print("using data:" + data_filename, flush=True)

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

    data_seed = args['data_seed']
    data = datasets[f"dataset_{data_seed}"]

    states = data["states"]
    actions = data["actions"]
    # print(f"actions averages: {np.mean(actions,axis=0)}", flush=True)
    print(f"state_dim: {states.shape}", flush=True)
    dataset = CustomDataset(states, actions)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # load parameters from config
    action_space_dim = config["file_attrs"]["A"]
    state_space_dim = config["file_attrs"]["K"]
    num_agents = config["file_attrs"]["N"]

    new_info = {"num_agents": num_agents,
                "state_space_dim": state_space_dim,
                "action_space_dim": action_space_dim
                }

    # model_name input argument should start with model name
    model_name = args['model_name']

    n_hidden_layers = args['n_hidden_layers']

    solver_dict = {"model_name": model_name,
                    "n_hidden_layers": n_hidden_layers,
                    "num_abs_agents": args['M'],
                    "abs_action_space_dim": args['L'],
                    "n_features": args['n_features'],
                    "num_agents": num_agents,
                    "state_space_dim": state_space_dim,
                    "action_space_dim": action_space_dim,
                    "enc2dec_ratio": args['enc2dec_ratio'],
                    "num_parameters": args['P'],
                     }
    print(args['P'])

    hidden_dim = get_width(solver_dict)
    hidden_dim = int(hidden_dim)
    print(f"hidden_dim={hidden_dim}")
    args['hidden_dim'] = hidden_dim  #not in hash
    if model_name == 'single':
        net = MultiChannelNet(
            n_channels=num_agents,
            input_size=state_space_dim,
            hidden_layer_width= hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_size=action_space_dim
        )
    else:
        print('choose valid model', flush=True)
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()  # takes logits
    # criterion = nn.BCEWithLogitsLoss() #since actions are binary

    # log number of parameters
    num_parameters = count_parameters(net)
    print(f"number of parameters: {num_parameters}", flush=True)
    print("gap between P and num_parameters: ", args['P'] - num_parameters, flush=True)


    num_action_samples = len(train_loader)*batch_size*num_agents

    # Access the existing config and add additional parameters
    new_info.update({
                    "num_action_samples": num_action_samples,
                    "num_parameters": num_parameters
                    })

    # Log the updated config
    # wandb.config.update(new_info,allow_val_change=True)
    
    # evaluate pretraining loss
    pre_training_loss = 0
    pre_training_accuracy = 0

    for i, data_batch in enumerate(train_loader, 0):
        inputs, labels = data_batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        action_logit_vectors = net(inputs)
        max_scores, max_idx_class = action_logit_vectors.max(dim=2)

        pre_training_loss += sum(criterion(
            torch.squeeze(
                action_logit_vectors[:, agent_idx, :]), 
                labels[:, agent_idx]
                ) 
            for agent_idx in range(num_agents)).item()
        pre_training_accuracy += (labels == max_idx_class).sum().item()
    print(
        f"pre training loss: {pre_training_loss/num_action_samples},"+\
        f" acc: {pre_training_accuracy/num_action_samples}", 
        flush=True)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    if args['use_lr_scheduler']:
        step_size = args['step_LR']
        gamma = args['gamma']
        # stop at learning_rate * (gamma ** (N // step_size)),
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) 
    logging_loss = []
    logging_acc = []

    # wandb.watch(net, log="all")
    
    # training loop
    for epoch in range(epochs):

        running_loss = 0.0
        running_correct = 0.0
        for i, (inputs, targets) in enumerate(train_loader, 0):

            inputs = inputs.to(device)
            labels = labels.to(device)
            # output is (batchsize, number of agents, action space size)
            action_logit_vectors = net(inputs)
            
            optimizer.zero_grad()
            loss = sum(criterion(torch.squeeze(
                action_logit_vectors[:, agent_idx, :]), labels[:, agent_idx]) 
                for agent_idx in range(num_agents))
            loss.backward()
            optimizer.step()

            if args['use_lr_scheduler']:
                scheduler.step()

            running_loss += loss.item()
            max_scores, max_idx_class = action_logit_vectors.max(dim=-1)
            running_correct += (labels == max_idx_class).sum().item()

        epoch_accuracy = running_correct / num_action_samples
        epoch_loss = running_loss / num_action_samples
        print(f"Epoch {epoch+1}, loss: {epoch_loss:.8}, acc: {epoch_accuracy:.8}", flush=True)
        # wandb.log({
        #     "epoch": epoch+1,
        #     "epoch_accuracy": epoch_accuracy, 
        #     "epoch_loss": epoch_loss,
        #     })
        logging_loss.append(epoch_loss)
        logging_acc.append(epoch_accuracy)

        # save model every log_interval epochs
        if (epoch+1) % log_interval == 0:
            torch.save(net.state_dict(), train_info_dir + f"/state_dict_{epoch+1}.pt")
            print(f"~saved model for epoch: {epoch+1}", flush=True)

    training_data = {}
    training_data['loss'] = logging_loss
    training_data['accuracy'] = logging_acc

    training_run_info = \
        f"_{args['model_name']}"+\
        f"_modelsize_{args['P']}"+\
        f"_trainseed_{seed}"+\
        f"_epochs_{epochs}"+\
        f"_batchsize_{batch_size}"+\
        f"_lr_{args['learning_rate']}"
    print("saving " + training_run_info + " at hash:", flush=True)
    print(hash_var, flush=True)

    # save training results dict as numpy
    np.save(train_info_dir + "/results.npy", training_data)

    # also save args as yaml
    with open(train_info_dir + "/args.yaml", 'w') as file:
        yaml.dump(args, file)

    torch.save(net.state_dict(), train_info_dir + "/state_dict_final.pt")

    return hash_var

# if __name__ == '__main__':
#     train(vars(args))