from torch.utils.data import Dataset, DataLoader
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Sampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from STOMPnet import STOMPnet
from utils import MultiChannelNet
import warnings
# import wandb
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--model_name', type=str,
                    default='STOMPnet_M_2_L_4_nfeatures_2', help='Name of the model')
                    # default='singletaskbaseline', help='Name of the model')
                    # default='multitaskbaseline', help='Name of the model')
parser.add_argument('--hidden_capacity', type=int,
                    default=240, help='capacity of abstract joint policy space')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--learning_rate', type=float,
                    default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--outdir', type=str, default='output/',
                    help='Output directory')
parser.add_argument('--data_filename', type=str,
                    default='_4agentdebug_modelname_bitpop_corr_1.0_ensemble_sum_M_2_simulationdata_actsel_greedy_numepi_1_K_10_N_4_T_10000_g_9.0', help='Data filename')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--data_seed', type=int,
                    default=0, help='data realization')

args = parser.parse_args()

# wandb.init(project='STOMP', entity='main_training')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device", flush=True)

seed: int = args.seed
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


if __name__ == '__main__':

    # training setting
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    training_paras = {}
    training_paras['epochs'] = epochs
    training_paras['learning_rate'] = learning_rate
    training_paras['batch_size'] = batch_size
    training_paras['train_seed'] = seed
    print(f"seed {seed} training of {args.model_name} with capacity {args.hidden_capacity} for {epochs} epochs using batchsize {batch_size} and LR {args.learning_rate}")

    # load the data from the output folder
    outdir = args.outdir
    data_settings = np.load(outdir + args.data_filename +
                            '.npy', allow_pickle=True).item()
    data_filename = args.data_filename + f'_dataseed_{args.data_seed}'
    print("using data:"+data_filename)
    data = np.load(outdir + data_filename + '.npy',
                   allow_pickle=True).item()
    states = data["states"]
    actions = data["actions"]

    dataset = CustomDataset(states, actions)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # load system parameters
    action_space_dim = 2
    state_space_dim = data_settings['sys_parameters']['K']
    num_agents = data_settings['sys_parameters']['N']

    # instantiate model
    model_paras = {}
    model_paras['hidden_capacity'] = args.hidden_capacity
    # model_name should end with list of parameter names and values
    model_paras['model_name'] = args.model_name.split('_')[0]
    if len(args.model_name.split('_')) > 1:
        para_list = args.model_name.split('_')[1:]
        for para_idx in range(int(len(para_list)/2)):
            model_paras[para_list[2*para_idx]] = int(para_list[2*para_idx+1])

    if model_paras['model_name'] == 'STOMPnet':
        # to match the M generated by bitpop set as: data_settings['sys_parameters']['jointagent_groundmodel_paras']['M']
        num_abs_agents = int(args.model_name.split('_')[2])
        abs_action_space_dim = int(args.model_name.split('_')[4])
        agent_embedding_dim = int(args.model_name.split('_')[6])
        assert (args.hidden_capacity/num_abs_agents).is_integer(
        ), "num of abstract agents should divide hidden dimensions"
        enc_hidden_dim = int(args.hidden_capacity/num_abs_agents)
        net = STOMPnet(
            state_space_dim,
            abs_action_space_dim,
            enc_hidden_dim,
            num_agents,
            num_abs_agents,
            action_space_dim=action_space_dim
        )
    elif model_paras['model_name'] == 'singletaskbaseline':
        assert (args.hidden_capacity /
                num_agents).is_integer(), "num of agents should divide hidden dimensions"
        net = MultiChannelNet(
            n_channels=num_agents,
            input_size=state_space_dim,
            hidden_layer_width=int(args.hidden_capacity/num_agents),
            output_size=action_space_dim
        )
    elif model_paras['model_name'] == 'multitaskbaseline':
        net = MultiChannelNet(
            n_channels=1,
            input_size=state_space_dim,
            hidden_layer_width=args.hidden_capacity,
            output_size=num_agents*action_space_dim,
            output_dim=(num_agents, action_space_dim)
        )
    else:
        print('choose valid model')
    net.to(device)

    criterion = nn.CrossEntropyLoss()  # takes logits
    # criterion = nn.BCEWithLogitsLoss() #since actions are binary

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    logging_loss = []
    logging_acc = []
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0.0
        for i, data_batch in enumerate(train_loader, 0):
            inputs, labels = data_batch
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            # output is (batchsize, number of agents, action space size)
            action_logit_vectors = net(inputs)
            # sum loss over agents (criterion function sums loss over samples in batch)
            loss = sum(criterion(torch.squeeze(action_logit_vectors[:, agent_idx, :]), labels.type(
                torch.LongTensor)[:, agent_idx]) for agent_idx in range(num_agents))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            max_scores, max_idx_class = net(inputs).max(dim=2)
            running_correct += (labels == max_idx_class).sum().item()
        epoch_accuracy = running_correct / \
            (len(train_loader)*batch_size*num_agents)
        epoch_loss = running_loss/(len(train_loader)*batch_size*num_agents)

        print(f"Epoch {epoch+1}, loss: {epoch_loss:.5}, acc: {epoch_accuracy:.5}", flush=True)
        logging_loss.append(epoch_loss)
        logging_acc.append(epoch_accuracy)
        # wandb.log({"epoch": epoch+1, "loss": epoch_loss})

    training_data = {}
    training_data['loss'] = logging_loss
    training_data['accuracy'] = logging_acc

    store_dict = {}
    store_dict['model_paras'] = model_paras
    store_dict['training_paras'] = training_paras
    store_dict['data_path'] = outdir + data_filename
    store_dict['training_data'] = training_data
    training_run_info = f"_{args.model_name}_cap_{args.hidden_capacity}_trainseed_{seed}_epochs_{epochs}_batchsz_{batch_size}_lr_{args.learning_rate}"
    print("saving " + training_run_info)
    np.save(outdir + data_filename +
            training_run_info + ".npy", store_dict)

    torch.save(net.state_dict(), outdir +
               data_filename + training_run_info + "_state_dict.pt")
