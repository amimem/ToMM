from torch.utils.data import Dataset, DataLoader
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from STOMPnet import STOMPnet
from utils import MultiChannelNet, count_parameters, get_width
import warnings
import os
import h5py
import yaml
import hashlib
import wandb
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--model_name', type=str,
                    # default='stomp', help='Name of the model')
                    default='single', help='Name of the model')
                    # default='multi', help='Name of the model')
parser.add_argument('--P', type=float, default=1e7,
                    help='Number of model parameters')
parser.add_argument('--M', type=int, default=2,
                    help='Number of abstract agents')
parser.add_argument('--L', type=int, default=100,
                    help='Abstract action space dimension')
parser.add_argument('--n_hidden_layers', type=int, default=2,
                    help='Number of hidden layers')
parser.add_argument('--n_features', type=int, default=2,
                    help='Agent embedding dimension')
parser.add_argument('--num_codebooks', type=int,
                    default=10, help='Number of codebooks')
parser.add_argument('--enc2dec_ratio', type=float,
                    default=1., help='Encoder to decoder ratio')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--learning_rate', type=float,
                    default=5e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--outdir', type=str, default='output/',
                    help='Output directory')
parser.add_argument('--data_dir', type=str,
                    default='data_e72dd17cbc/', help='Data directory')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--data_seed', type=int,
                    default=0, help='data realization')
parser.add_argument('--checkpoint_interval', type=int, default=100, help='Checkpointing interval')

args = parser.parse_args()

# logging to wandb
wandb.login()

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
    log_interval = args.checkpoint_interval

    print(f"seed {seed} training of {args.model_name} with modelsize {args.P} for {epochs} epochs using batchsize {batch_size} and LR {args.learning_rate}")

    # load the data from the output folder
    outdir = args.outdir
    data_dir = args.data_dir
    data_filename = os.path.join(outdir, data_dir, 'data.h5')
    config_filename = os.path.join(outdir, data_dir, 'config.yaml')
    
    # make a directory for saving the results
    save_dir = os.path.join(outdir, data_dir, 'training_results/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # hash the arguments except for the outdir
    hash_dict = args.__dict__.copy()
    hash_dict.pop('outdir')

    # make a subdirectory for each run
    hash = hashlib.blake2s(str(hash_dict).encode(), digest_size=5).hexdigest()
    train_info_dir = os.path.join(save_dir, hash)
    if not os.path.exists(train_info_dir):
        os.makedirs(train_info_dir)

    # initialize wandb
    wandb.init(project="STOMP", name=hash, config=args.__dict__)

    print("using data:" + data_filename)

    # load the hdf data
    with h5py.File(data_filename, 'r') as f:
        datasets = {}
        for group_name, group in f.items():
            datasets[group_name] = {key: np.array(
                value) for key, value in group.items()}

    # load the config file
    with open(config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)

    data_seed = args.data_seed
    data = datasets[f"dataset_{data_seed}"]

    states = data["states"]
    actions = data["actions"]

    print(f"actions averages: {np.mean(actions,axis=0)}")

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
    model_name = args.model_name
    n_hidden_layers = args.n_hidden_layers

    solver_dict = {"model_name": model_name,
                    "n_hidden_layers": n_hidden_layers,
                    "num_abs_agents": args.M,
                    "abs_action_space_dim": args.L,
                    "n_features": args.n_features,
                    "num_agents": num_agents,
                    "state_space_dim": state_space_dim,
                    "action_space_dim": action_space_dim,
                    "enc2dec_ratio": args.enc2dec_ratio,
                    "num_parameters": args.P,
                     }

    hidden_dim = get_width(solver_dict)
    hidden_dim = int(hidden_dim)

    if model_name == 'stomp':
        # to match the M generated by joint policy of ground model set as: data_settings['sys_parameters']['jointagent_groundmodel_paras']['M']
        net = STOMPnet(
            state_space_dim = state_space_dim,
            abs_action_space_dim = args.L,
            enc_hidden_dim = hidden_dim,
            dec_hidden_dim = int(args.enc2dec_ratio*hidden_dim),
            num_agents=num_agents,
            num_abs_agents = args.M,
            action_space_dim=action_space_dim,
            agent_embedding_dim= args.n_features,
            n_hidden_layers=n_hidden_layers,
            num_codebooks=args.num_codebooks
        )
    elif model_name == 'single':
        net = MultiChannelNet(
            n_channels=num_agents,
            input_size=state_space_dim,
            hidden_layer_width= hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_size=action_space_dim
        )
    elif model_name == 'multi':
        net = MultiChannelNet(
            n_channels=1,
            input_size=state_space_dim,
            hidden_layer_width= hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_size=num_agents*action_space_dim,
            output_dim=(num_agents, action_space_dim)
        )
    else:
        print('choose valid model')
    net.to(device)

    criterion = nn.CrossEntropyLoss()  # takes logits
    # criterion = nn.BCEWithLogitsLoss() #since actions are binary

    # log number of parameters
    num_parameters = count_parameters(net)
    print(f"number of parameters: {num_parameters}")
    print("gap between P and num_parameters: ", args.P - num_parameters)


    num_action_samples = len(train_loader)*batch_size*num_agents

    # Access the existing config and add additional parameters
    new_info.update({
                    "num_action_samples": num_action_samples,
                    "num_parameters": num_parameters
                    })

    # Log the updated config
    wandb.config.update(new_info)
    
    # evaluate pretraining loss
    pre_training_loss = 0
    pre_training_accuracy = 0
    for i, data_batch in enumerate(train_loader, 0):
        inputs, labels = data_batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        action_logit_vectors = net(inputs)
        max_scores, max_idx_class = action_logit_vectors.max(dim=2)

        pre_training_loss += sum(criterion(torch.squeeze(
            action_logit_vectors[:, agent_idx, :]), labels[:, agent_idx]) for agent_idx in range(num_agents)).item()
        pre_training_accuracy += (labels == max_idx_class).sum().item()
    print(f"pre training loss: {pre_training_loss/num_action_samples}, acc: {pre_training_accuracy/num_action_samples}")

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    logging_loss = []
    logging_acc = []
    last_loss = 0


    # training loop
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0.0
        for i, data_batch in enumerate(train_loader, 0):
            inputs, labels = data_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # output is (batchsize, number of agents, action space size)
            action_logit_vectors = net(inputs)

            loss = sum(criterion(torch.squeeze(
                action_logit_vectors[:, agent_idx, :]), labels[:, agent_idx]) for agent_idx in range(num_agents))
            # if model_paras['model_name'] == 'STOMPnet':
            #     # add l1 regularization of assigner probabilities
            #     assigner_logits = net.state_dict(
            #     )["assigner.abs_agent_assignment_embedding.weight"]
            #     assigner_probs = F.softmax(assigner_logits,dim=-1)
            #     lambda1 = 10
            #     l1_regularization_of_assigner_probs = sum(lambda1 * \
            #         torch.norm(assigner_probs, p=1,dim=-1))
            #     loss = loss + l1_regularization_of_assigner_probs

            loss.backward()

            # norms=torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.01)

            optimizer.step()

            running_loss += loss.item()
            max_scores, max_idx_class = action_logit_vectors.max(dim=-1)
            running_correct += (labels == max_idx_class).sum().item()
            wandb.watch(net, log="all")

        epoch_accuracy = running_correct / num_action_samples
        epoch_loss = running_loss / num_action_samples
        last_loss = epoch_loss
        print(f"Epoch {epoch+1}, loss: {epoch_loss:.8}, acc: {epoch_accuracy:.8}", flush=True)
        logging_loss.append(epoch_loss)
        logging_acc.append(epoch_accuracy)

        # save model every log_interval epochs
        if (epoch+1) % log_interval == 0:
            torch.save(net.state_dict(), train_info_dir + f"/state_dict_{epoch+1}.pt")
            print(f"~saved model for epoch: {epoch+1}", flush=True)

        # log to wandb
        wandb.log({"epoch": epoch+1,
                   "epoch_accuracy": epoch_accuracy, 
                   "epoch_loss": epoch_loss,
                   })

    training_data = {}
    training_data['loss'] = logging_loss
    training_data['accuracy'] = logging_acc

    training_run_info = f"_{args.model_name}_modelsize_{args.P}_trainseed_{seed}_epochs_{epochs}_batchsize_{batch_size}_lr_{args.learning_rate}"
    print("saving " + training_run_info)

    # save training results dict as numpy
    np.save(train_info_dir + "/results.npy", training_data)

    # also save args as yaml
    with open(train_info_dir + "/args.yaml", 'w') as file:
        yaml.dump(args.__dict__, file)

    torch.save(net.state_dict(), train_info_dir + "/state_dict_final.pt")