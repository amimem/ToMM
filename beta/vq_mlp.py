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

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost = 0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.num_codes = None
        self.reset_parameters()

    def forward(self, latents):
        # latent shape: (batch_size, z)
        # reshape latents to (batch_size, -1, D_e)
        latents = latents.reshape(latents.shape[0], -1, self.embedding.weight.shape[-1])
        self.num_codes = latents.shape[1]

        # log the l2 norm of the latents
        wandb.log({"l2_norm_latents": torch.linalg.norm(latents, ord=2, dim=-1).mean().item()})
        # embedding shape: (N_e, D_e)
        # Compute L2 distances between latents and embedding weights
        weights = self.embedding.weight # weights shape: (N_e, D_e)
        sub = latents.unsqueeze(-2) - weights # latents shape: (batch_size, z//D_e , 1, D_e), weights shape: (N_e, D_e)
        dist = torch.linalg.vector_norm(sub, dim=-1) # dist shape: (batch_size, z//D_e, N_e)
        encoding_inds = torch.argmin(dist, dim=-1)        # Get the number of the nearest codebook vector, shape: (batch_size, z//D_e)
        quantized_latents = self.quantize(encoding_inds)  # Quantize the latents of shape (batch_size, z//D_e, D_e)

        # log the l2 norm of the quantized latents
        wandb.log({"l2_norm_quantized_latents": torch.linalg.norm(quantized_latents, ord=2, dim=-1).mean().item()})

        # Compute the VQ Losses
        codebook_loss = F.mse_loss(latents.detach(), quantized_latents)
        commitment_loss = F.mse_loss(latents, quantized_latents.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # get a new version loss as row wise dot product of latents and quantized latents
        # dot_loss = torch.sum(latents*quantized_latents, dim=-1) # shape: (batch_size)
        # softmax_loss = F.softmax(dot_loss, dim=-1) # shape: (batch_size)
        # vq_loss = torch.sum(softmax_loss)
        
        # Make the gradient with respect to latents be equal to the gradient with respect to quantized latents 
        quantized_latents = latents + (quantized_latents - latents).detach()
        # reshape quantized_latents to (batch_size, -1) where -1 = z
        quantized_latents = quantized_latents.reshape(quantized_latents.shape[0], quantized_latents.shape[-1]*quantized_latents.shape[-2])
        return quantized_latents, vq_loss
    
    def quantize(self, encoding_indices):
        z = self.embedding(encoding_indices) # z shape: (batch_size, z//D_e)
        return z
    
    def reset_parameters(self):
        # use the default normal initialization
        pass


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

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_embeddings=12, embedding_dim = 16, vq_projection_dim = 64, num_hidden_layers=2, commitment_cost=0.25):
        super(MLP, self).__init__()

        self.z = vq_projection_dim
        self.num_embeddings = num_embeddings

        # assert that z is divisble by embedding_dim, if not, increase z
        if self.z % embedding_dim != 0:
            self.z = (self.z // embedding_dim + 1) * embedding_dim
            print(f"z is not divisible by embedding_dim, increasing z to {self.z}")

        # make sure num_embeddings is a even
        if num_hidden_layers % 2 != 0:
            num_hidden_layers += 1
        
        half_num_hidden_layers = num_hidden_layers // 2

        self.input_layer = nn.Linear(input_size, hidden_size) # input_size: sequence_length * (state_dim + num_actions)
        self.hidden_layer_1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(half_num_hidden_layers)])
        self.hidden_project_in = nn.Linear(hidden_size, self.z)
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.hidden_project_out = nn.Linear(self.z, hidden_size)
        self.hidde_layer_2 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(half_num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.activation(self.input_layer(x))

        for layer in self.hidden_layer_1:
            x = layer(x)
            # x = self.layer_norm(x)
            x = self.activation(x)

        x = self.hidden_project_in(x)
        x = self.activation(x)

        x, vq_loss = self.vq(x)

        x = self.hidden_project_out(x)
        x = self.activation(x)

        for layer in self.hidde_layer_2:
            x = layer(x)
            # x = self.layer_norm(x))
            x = self.activation(x)

        # x = self.layer_norm(x)
        x = self.output_layer(x)

        return x, vq_loss

# create a Dataset object
class CustomDataset(Dataset):
    def __init__(self, states, actions, sequence_length, seed=0, train=True):
        # get a rng
        self.rng = np.random.RandomState(seed)
        self.train = train
        self.states = [torch.tensor(states[i:i+sequence_length]).float() for i in range(len(states) - sequence_length + 1)]
        self.actions = [torch.tensor(actions[i:i+sequence_length]).long() for i in range(len(actions) - sequence_length + 1)]
        self.sequence_length = sequence_length
        self.num_agents = actions.shape[1]
        self.train_agents = self.rng.randint(0, self.num_agents, size=len(self.states))
        self.validation_agents = self.rng.randint(0, self.num_agents, size=len(self.states))
        for i, (t,v) in enumerate(zip(self.train_agents, self.validation_agents)):
            if t == v:
                self.validation_agents[i] = (v + 1) % self.num_agents

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # random sample x from the number of agents
        if self.train:
            split = self.train_agents
        else:
            split = self.validation_agents
        return self.states[idx], self.actions[idx][:, split[idx]]
    
def train(model, dataloader, num_actions=2):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    batch_loss = []
    batch_vq_loss = []
    batch_accuracy = []
    for i, (state, action) in enumerate(dataloader):
        optimizer.zero_grad()
        # bsz, seqlen, statedim = state.shape
        # bsz, seqlen = action.shape
        state = state.flatten(start_dim=1)
        action_onehot = torch.nn.functional.one_hot(action, num_classes=num_actions).flatten(start_dim=1) # (batch_size, sequence_length * num_actions)
        action_onehot[:, -num_actions:] = 0 # set the last num_actions to 0, the network should predict the last action
        state = torch.hstack([state, action_onehot.float()]) 

        output, vq_loss = model(state)

        # check the percentage of the logits that are negative
        wandb.log({"percentage_negative_logits": torch.sum(output < 0).item() / output.numel()})
        # log the histogram of the logits
        wandb.log({"logits_histogram": wandb.Histogram(output.flatten().detach().cpu().numpy())})

        agent_action = action[:, -1]
        loss = criterion(output, agent_action) \
            + vq_loss
        accuracy = (torch.argmax(output, dim=1) == agent_action).float().mean().item()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        batch_vq_loss.append(vq_loss.item())
        batch_accuracy.append(accuracy)

        wandb.log({"batch_loss": loss.item(), "batch_accuracy": accuracy, "batch_vq_loss": vq_loss.item()})
        
    return np.mean(batch_loss), np.mean(batch_accuracy), np.mean(batch_vq_loss)

def test(model, dataloader, num_actions=2):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    batch_loss = []
    batch_vq_loss = []
    batch_accuracy = []
    for i, (state, action) in enumerate(dataloader):
        # bsz, seqlen, statedim = state.shape
        # bsz, seqlen = action.shape
        state = state.flatten(start_dim=1)
        action_onehot = torch.nn.functional.one_hot(action, num_classes=num_actions).flatten(start_dim=1)
        action_onehot[:, -num_actions:] = 0
        state = torch.hstack([state, action_onehot.float()])

        output, vq_loss = model(state)
        agent_action = action[:, -1]
        loss = criterion(output, agent_action) \
            + vq_loss
        accuracy = (torch.argmax(output, dim=1) == agent_action).float().mean().item()
        batch_loss.append(loss.item())
        batch_vq_loss.append(vq_loss.item())
        batch_accuracy.append(accuracy)
        
    return np.mean(batch_loss), np.mean(batch_accuracy), np.mean(batch_vq_loss)


def get_data_loader(data, sequence_length):
    states = data["states"]
    actions = data["actions"]
    # shape of states: (num_states, dim_teacher_inp)
    # shape of actions: (num_states, num_teachers)
    print(states.shape, actions.shape)

    train_dataset = CustomDataset(states, actions, sequence_length, seed=seed, train=True)
    test_dataset = CustomDataset(states, actions, sequence_length, seed=seed, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_dataloader, test_dataloader

def get_model(dataloader, num_actions: int, hidden_size=64, num_groups=2, num_hidden_layers=2, vq_embedding_dim=16):

    state, action = next(iter(dataloader))
    # state shape: (batch_size, sequence_length, state_dim)
    # action shape: (batch_size, sequence_length)
    state = state.flatten(start_dim=1) # (batch_size, sequence_length * state_dim)
    action_onehot = torch.nn.functional.one_hot(action, num_classes=num_actions).flatten(start_dim=1) # (batch_size, sequence_length * num_actions)
    state = torch.hstack([state, action_onehot.float()]) # (batch_size, sequence_length * (state_dim + num_actions))

    # Create an instance of the MLP
    input_size = state.shape[1] # sequence_length * (state_dim + num_actions)

    # find the closest power 2 to the number of groups (upper bound)
    # num_embeddings = 2**np.ceil(np.log2(num_groups) + 1)
    num_embeddings = int(hidden_size)
    print(f"Number of embeddings: {num_embeddings}")

    # solve z for the equation: 2*hidden_size*z + num_embeddings*z = hidden_size**2
    # z = int(hidden_size**2 // (2*hidden_size + num_embeddings))
    z = hidden_size

    mlp = MLP(input_size, hidden_size, num_actions, num_embeddings, vq_embedding_dim, z, num_hidden_layers)      

    return mlp

if __name__ == "__main__":
    set_seed(seed)

    # for n_agents and n_groups = n_agents
    data_hashes = ["data_6013b64ce8",]
                #    "data_dfdaecc3ee",]
                    # "data_8646a4bdd8", "data_e94dbedcea",\
                    # "data_813427ec72", "data_fbe4154fff", "data_d4fcf6cdef", "data_8950a6aae5",\
                    # "data_97b6c3ab33", "data_d8fd9e8472", "data_fcb20c7d4a", "data_e8cbc57b61",\
                    # "data_f714057b40", "data_567898bdec", "data_f2b68367cd", "data_d4588ac462"] \
                #     + \
                #    ["data_6013b64ce8", "data_27dea4bcce", "data_41f896f2be", "data_9bd5f5ee5f",\
                #     "data_4af6f9d879", "data_a71679fd65", "data_46ef7fc2a7", "data_935be5ac8f",\
                #     "data_eb9b7315c6", "data_0b7d25ce95", "data_89a277ffd9", "data_cf84771ef1",\
                #     "data_76a571ea15", "data_dbf79f7d01", "data_5f734fb2a1", "data_6644bb7ada"]
    
    
    # data_hashes = ["data_6013b64ce8", "data_27dea4bcce", "data_41f896f2be", "data_9bd5f5ee5f",\
    #                 "data_4af6f9d879", "data_a71679fd65", "data_46ef7fc2a7", "data_935be5ac8f",\
    #                 "data_eb9b7315c6", "data_0b7d25ce95", "data_89a277ffd9", "data_cf84771ef1",\
    #                 "data_76a571ea15", "data_dbf79f7d01", "data_5f734fb2a1", "data_6644bb7ada"] # for n_agents and n_groups = 1
    
    # make sure all data_hashes are in the output folder
    assert all([os.path.exists(f"output/{data_hash}") for data_hash in data_hashes])
    print("All data hashes are in the output folder")

    sequence_lengths = [8]
    w_d = [(256,2)]
    vq_embedding_dim = 4
    num_epochs = 10

    df = pd.DataFrame(columns=[
        "data_hash", "num_agents", "num_groups", "state_dim", "vq_projection_dim", "num_actions", "sequence_length",
          "hidden_size", "num_hidden_layers", "epoch", "train_loss", "train_accuracy",
            "test_loss", "test_accuracy", "train_vq_loss", "test_vq_loss"])
    # sequence_lengths = [sequence_lengths[job_id]]
    # print(f"Running sequence length {sequence_lengths}")

    if job_id != -1:
        data_hashes = [data_hashes[job_id]]

    for data_hash in data_hashes:
        for sequence_length in sequence_lengths:
            for hidden_size, num_hidden_layers in w_d:
                data, config = load_data(data_hash)
                num_actions = config["file_attrs"]["num_actions"]
                num_groups = config["file_attrs"]["num_groups"]
                train_dataloader, test_dataloader = get_data_loader(data, sequence_length)
                mlp = get_model(train_dataloader, num_actions, hidden_size, num_groups, num_hidden_layers=num_hidden_layers, vq_embedding_dim=vq_embedding_dim)
                vq_projection_dim = mlp.z
                num_embeddings = mlp.num_embeddings
                num_codes = mlp.vq.num_codes if mlp.vq.num_codes is not None else mlp.z//vq_embedding_dim

                wandb_run_name = f"sl_{sequence_length}_wd_{w_d}_z_{vq_projection_dim}_edim_{vq_embedding_dim}_ne_{num_embeddings}_nc_{num_codes}"
                print(f"Running {wandb_run_name}")

                config.update({"sequence_length": sequence_length, "hidden_size": hidden_size, "num_hidden_layers": num_hidden_layers,
                                "vq_embedding_dim": vq_embedding_dim, "vq_projection_dim": vq_projection_dim, "num_actions": num_actions})
                wandb.init(project="MLP", group="May_10th_vq_test", job_type=None, config=config, name=wandb_run_name)

                for epoch in range(num_epochs):
                    # train the model
                    train_epoch_loss, train_epoch_accuracy, train_epoch_vq_loss = train(mlp, train_dataloader, num_actions=num_actions)
                    # test the model
                    test_epoch_loss, test_epoch_accuracy, test_epoch_vq_loss = test(mlp, test_dataloader, num_actions=num_actions)
                    # print both the train and test epoch loss and accuracy and vq_loss
                    print(f"Epoch: {epoch}, Train Loss: {train_epoch_loss}, Train Accuracy: {train_epoch_accuracy}, Train VQ Loss: {train_epoch_vq_loss}")
                    print(f"Epoch: {epoch}, Test Loss: {test_epoch_loss}, Test Accuracy: {test_epoch_accuracy}, Test VQ Loss: {test_epoch_vq_loss}")

                    wandb.log({"epoch": epoch, "train_loss": train_epoch_loss, "train_accuracy": train_epoch_accuracy, \
                                "test_loss": test_epoch_loss, "test_accuracy": test_epoch_accuracy, \
                                    "train_vq_loss": train_epoch_vq_loss,
                                        "test_vq_loss": test_epoch_vq_loss})
                    # Append the data to the DataFrame
                    new_row = pd.DataFrame({
                        "data_hash": [data_hash],
                        "num_agents": [config["file_attrs"]["num_teachers"]],
                        "num_groups": [config["file_attrs"]["num_groups"]],
                        "state_dim": [config["file_attrs"]["dim_state"]],
                        "vq_projection_dim": [vq_projection_dim],
                        "num_actions": [num_actions],
                        "sequence_length": [sequence_length],
                        "hidden_size": [hidden_size],
                        "num_hidden_layers": [num_hidden_layers],
                        "epoch": [epoch],
                        "train_loss": [train_epoch_loss],
                        "train_accura   cy": [train_epoch_accuracy],
                        "test_loss": [test_epoch_loss],
                        "test_accuracy": [test_epoch_accuracy],
                        "train_vq_loss": [train_epoch_vq_loss],
                        "test_vq_loss": [test_epoch_vq_loss]
                    }, index=[0])
                    df = pd.concat([df, new_row], ignore_index=True)
                wandb.finish()

    time_str = time.strftime("%Y-%m-%d-%H-%M")
    df.to_csv(f"output/{time_str}_results.csv", index=False)