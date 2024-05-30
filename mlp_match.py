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
from models import Model, MLP_baseline
from data_utils import load_data, get_seqdata_loaders

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
        action_logit_vectors = model(state_seq, action_seqs)

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


def train(data_hash, config):
    config = SimpleNamespace(**config)

    data, data_config = load_data(data_hash)
    data_config = SimpleNamespace(**data_config['file_attrs'])
    config.data_config = data_config

    train_dataloader, test_dataloader = get_seqdata_loaders(
        data,
        config.seq_len,
        num_actions=data_config.A,
        train2test_ratio=config.train2test_ratio,
        batch_size=config.batch_size,
    )
    config.context_size = train_dataloader.dataset.context_size
    config.num_actions = train_dataloader.dataset.num_actions
    config.num_agents = train_dataloader.dataset.num_agents
    config.state_dim = train_dataloader.dataset.state_dim

    # seq_lens = [seq_lens[job_id]]
    # print(f"Running context length {seq_lens}")
    # if job_id != -1:
    #     data_hashes = [data_hashes[job_id]]
    if config.model_name=='STOMP':
        model = STOMP(config)
    elif config.model_name=='MLP':
        model = MLP_baseline(config)
    elif config.model_name =='sharedMLP':
        model = sharedMLP_baseline(config)

    # log number of parameters
    num_parameters = model.count_parameters()
    print(f"number of parameters: {num_parameters}", flush=True)
    print("gap between P and num_parameters: ", config.P - num_parameters, flush=True)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # wandb.init(project="TOGM", group="archcompare",job_type=None, config=config)
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
            "num_agents": [data_config.N],
            "num_groups": [data_config.M],
            "state_dim": [data_config.K],
            "num_actions": [data_config.A],
            "decoder_type": [config.decoder_type],
            "seq_len": [config.seq_len],
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
    store_name = f"output/{time_str}_results.csv"
    df.to_csv(store_name, index=False)

    return store_name


if __name__ == "__main__":

    set_seed(seed)

    # experiment parameters
    N = 10
    K = 8
    corr = 1.
    P = 1e6

    # set parameters
    M_sys = 1
    M_train = 1
    datagen_epochs = 1
    datagen_P = 1e6
    epochs = 10 
    data_seed = 0
    num_samples = 10000
    M = 1

    # fixed parameters
    # read data
    hashlist_filename = f"hashlist_K_{K}_Msys_{M_sys}_T_{num_samples}_Mtrain_{M_train}_Ep_{datagen_epochs}_dataseed_{data_seed}.npy"
    hash_data = np.load(hashlist_filename,allow_pickle=True).item()
    df = hash_data['hashes']
    # print(df)
    # assert all([os.path.exists(f"output/{data_hash}") for data_hash in list(df.hash.values)])
    # print("All data /hashes are in the output folder")
    # set data
    # data_hash = df.loc[(df.corr == corr) & (df.N == N) & (
    # df.hashtype == 'bitpop_data') & (df.P == -1), 'hash'].values
    data_hash = 'data_d1bac3a730'
    # model and training configuration
    config = {
        'M_model': M,
        'seq_len': 16,
        'P': P,
        'num_epochs': epochs,
        'learning_rate': 5e-5,
        'batch_size': 8,
        'train2test_ratio': 0.8
        }

    # set architecture type:
    # >STOMP
    if False:
        config['model_name'] = 'STOMP'
        # config['enc_MLPhidden_dim'] = 256
        # config['enc_hidden_dim'] = 256
        # config['enc_out_dim'] = 256
        if True: # --single-agent baseline
            config['cross_talk'] = False
            config['decoder_type'] = 'MLP'
        else: # --multi-agent baseline
            config['cross_talk'] = True
            config['decoder_type'] = 'BuffAtt'
            # config['dec_hidden_dim'] = 256
    else:
        # >illustrative baselines
        # config['enc_out_dim'] = 256
        if True: # unshared
            config['model_name'] = 'MLP'
        else: # shared
            config['model_name'] = 'sharedMLP'
        config['cross_talk'] = None
        config['decoder_type'] = None

    store_name = train(data_hash, config)

    print(f'finished. output stored at: {store_name}')
