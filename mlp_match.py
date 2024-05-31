import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import wandb
import time
import os
from types import SimpleNamespace

# import custom functions
from models import STOMP, MLPperagent,sharedMLP,MLPallagents
from data_utils import load_data, get_seqdata_loaders, generate_dataset_from_logitmodel

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


# iterates over dataset, evaluating model response and optionally updating parameters
def eval_perf(model, dataloader, and_train=False, optimizer=None):
    if and_train:
        model.train()
    else:
        model.eval()

    criterion = nn.CrossEntropyLoss()

    batch_loss = []
    batch_accuracy = []
    for i, (state_seq, action_seqs, target_actions) in enumerate(dataloader):
        # agent_contexts: (bsz, num_agents, context_size), context_size = seq_len*(state_space_dim+num_actions)
        # targets: (bsz, num_agents)
        print(i)
        action_logit_vectors = model(state_seq, action_seqs)

        if hasattr(model, "decoder_type"):
            if model.decoder_type == 'BuffAtt' and i==0:
                # waste first batch to populate buffer
                model.decoder.append_to_buffer(target_actions)
                continue

        if and_train:
            optimizer.zero_grad()

        loss = sum(
            criterion(
                action_logit_vectors[:, agent_idx, :], target_actions[:, agent_idx])
            for agent_idx in range(dataloader.dataset.num_agents)
        )

        if and_train:
            loss.backward()
            optimizer.step()

        accuracy = (torch.argmax(action_logit_vectors, dim=-1)
                    == target_actions).float().mean().item()
        batch_loss.append(loss.item())
        batch_accuracy.append(accuracy)

    return np.mean(batch_loss), np.mean(batch_accuracy)


def train(config):
    data, data_config = load_data(config['data_hash'],data_seed=config['data_seed'])
    data_config = SimpleNamespace(**data_config['file_attrs'])
    model_config = SimpleNamespace(**config['model_config'])
    config = SimpleNamespace(**config)
    # config.data_config = vars(data_config)

    train_dataloader, test_dataloader = get_seqdata_loaders(
        data,
        model_config.seq_len,
        num_actions=data_config.num_actions,
        train2test_ratio=config.train2test_ratio,
        batch_size=config.batch_size,
    )
    model_config.context_size = train_dataloader.dataset.context_size
    model_config.num_actions = train_dataloader.dataset.num_actions
    model_config.num_agents = train_dataloader.dataset.num_agents
    model_config.state_dim = train_dataloader.dataset.state_dim
    model_config.P = config.P
    # seq_lens = [seq_lens[job_id]]
    # print(f"Running context length {seq_lens}")
    # if job_id != -1:
    #     data_hashes = [data_hashes[job_id]]
    if model_config.model_name=='STOMP':
        model = STOMP(model_config)
    elif model_config.model_name=='MLPperagent':
        model = MLPperagent(model_config)
    elif model_config.model_name =='sharedMLP':
        model = sharedMLP(model_config)
    elif model_config.model_name =='MLPallagents':
        model = MLPallagents(model_config)

    # log number of parameters
    num_parameters = model.count_parameters()
    print(f"number of parameters: {num_parameters}", flush=True)
    print("gap between P and num_parameters: ", config.P - num_parameters, flush=True)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # wandb.init(project="TOGM", group="archcompare",job_type=None, config=vars(config))
    for epoch in range(config.num_epochs):
        st=time.time()
        train_epoch_loss, train_epoch_accuracy = eval_perf(
            model, train_dataloader, and_train=True, optimizer=optimizer)
        test_epoch_loss, test_epoch_accuracy = eval_perf(
            model, test_dataloader)

        print(
            f"Epoch {epoch+1}/{config.num_epochs}, " +
            f"Train/Test Loss: {train_epoch_loss:.4f}/{test_epoch_loss:.4f}, " +
            f"Accuracy: {train_epoch_accuracy:.4f}/{test_epoch_accuracy:.4f}, " +
            f"took {int(time.time()-st)}"
        )
        # wandb.log({
        #     "epoch": epoch,
        #     "train_loss": train_epoch_loss,
        #     "train_accuracy": train_epoch_accuracy,
        #     "test_loss": test_epoch_loss,
        #     "test_accuracy": test_epoch_accuracy
        # })

        # Append the data to the DataFrame
        new_row = pd.DataFrame({
            "data_hash": [data_hash],
            "num_agents": [model_config.num_agents],
            # "num_groups": [data_config.M],
            "state_dim": [model_config.state_dim],
            "num_actions": [model_config.num_actions],
            "decoder_type": [model_config.decoder_type],
            "seq_len": [model_config.seq_len],
            "epoch": [epoch],
            "train_loss": [train_epoch_loss],
            "train_accuracy": [train_epoch_accuracy],
            "test_loss": [test_epoch_loss],
            "test_accuracy": [test_epoch_accuracy]
        }, index=[0])
        df = pd.concat([df, new_row], ignore_index=True) if epoch>0 else new_row
    # wandb.finish()

    time_str = time.strftime("%Y-%m-%d-%H-%M")
    data_dir = f"output/{data_hash}"
    data_filename = f"{data_dir}"
    store_name = f"{data_filename}/{time_str}_results.csv"
    df.to_csv(store_name, index=False)

    return store_name


if __name__ == "__main__":

    set_seed(seed)

    # experiment parameters
    N = 10
    K = 8
    corr = 0.99
    P = 1e6

    #generate correlation-controlled (s,a)-tuple data
    dataset_config = {
        "num_samples": int(1e4),
        "output" : "output",
        "num_actions":2,
        "num_agents": N,
        "state_dim": K,
        "model_name":"logit",
        "corr": corr
    }
    data_hash=generate_dataset_from_logitmodel(dataset_config)
    # hash

    #----------------------------------

    # model and training configuration for (s,a) sequence data
    model_config = {'seq_len': 16}
    # set architecture type:
    # >STOMP
    if False:
        model_config['model_name'] = 'STOMP'
        # config['enc_MLPhidden_dim'] = 256
        # config['enc_hidden_dim'] = 256
        # config['enc_out_dim'] = 256
        if True: # --single-agent baseline
            model_config['cross_talk'] = False
            model_config['decoder_type'] = 'MLP'
        else: # --multi-agent baseline
            model_config['cross_talk'] = True
            model_config['decoder_type'] = 'BuffAtt'
            # config['dec_hidden_dim'] = 256
    else:
        # >illustrative baselines
        # config['enc_out_dim'] = 256
        if True: # unshared
            model_config['model_name'] = 'MLPperagent'
        elif False: # shared
            model_config['model_name'] = 'sharedMLP'
        elif False: # 1 network 
            model_config['model_name'] = 'MLPallgents'
        model_config['cross_talk'] = None
        model_config['decoder_type'] = None

    #set training parameters
    config = {
        'P': P,
        'num_epochs': 10,
        'learning_rate': 5e-5,
        'batch_size': 8,
        'train2test_ratio': 0.8,
        'data_hash': data_hash,
        'data_seed': 0,
        'model_config': model_config
        }

    store_name = train(config)

    print(f'finished. output stored at: {store_name}')
