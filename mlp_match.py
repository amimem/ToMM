import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import wandb
import time
import hashlib
import os
import yaml
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
    for batch_step, (state_seq, action_seqs, target_actions) in enumerate(dataloader):
        # state_seq: (batch_size, seq_len, state_dim)
        # action_seqs: (batch_size, seq_len, num_agents, num_actions)
        # target actions: (bsz, num_agents)
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

    data_dir = config['outdir'] + config['data_dir']
    data, data_config = load_data(data_dir,data_seed=config['data_seed'])
    data_config = SimpleNamespace(**data_config['file_attrs'])
    model_config = SimpleNamespace(**config['model_config'])
    config = SimpleNamespace(**config)

    # make a directory for saving the results
    save_dir = os.path.join(config.outdir, config.data_dir, 'training_results/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # hash the arguments except for the outdir
    hash_dict = vars(config).copy()
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


    train_dataloader, test_dataloader = get_seqdata_loaders(
        data,
        model_config.seq_len,
        num_actions=data_config.num_actions,
        train2test_ratio=config.train2test_ratio,
        batch_size=config.batch_size,
    )
    model_config.num_actions = train_dataloader.dataset.num_actions
    model_config.num_agents = train_dataloader.dataset.num_agents
    model_config.state_dim = train_dataloader.dataset.state_dim
    model_config.P = config.P
 
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
    print(f"seed {config.seed} training of {model_config.model_name} model " +\
        f"with modelsize {model_config.P} for {config.num_epochs} epochs " +\
        f"using batchsize {config.batch_size} and LR {config.learning_rate}", flush=True)

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
            "data_hash": [config.data_dir],
            "num_agents": [model_config.num_agents],
            "state_dim": [model_config.state_dim],
            "num_actions": [model_config.num_actions],
            "decoder_type": [model_config.decoder_type],
            "seq_len": [model_config.seq_len],
            "epoch": [epoch],
            "train_loss": [train_epoch_loss],
            "train_accuracy": [train_epoch_accuracy],
            "test_loss": [test_epoch_loss],
            "test_accuracy": [test_epoch_accuracy],
            "data_seed": [config.data_seed]
        }, index=[0])
        df = pd.concat([df, new_row], ignore_index=True) if epoch>0 else new_row
    # wandb.finish()

    training_run_info = \
        f"_{model_config.model_name}"+\
        f"_modelsize_{model_config.P}"+\
        f"_trainseed_{config.seed}"+\
        f"_epochs_{config.num_epochs}"+\
        f"_batchsize_{config.batch_size}"+\
        f"_lr_{config.learning_rate}"
    print("saving " + training_run_info + " at hash:", flush=True)
    print(hash_var, flush=True)

    # also save config as yaml
    with open(train_info_dir + "/model_config.yaml", 'w') as file:
        yaml.dump(model_config, file)

    torch.save(model.state_dict(), train_info_dir + "/state_dict_final.pt")

    # time_str = time.strftime("%Y-%m-%d-%H-%M")
    data_filename = f"{data_dir}"
    df.to_csv(train_info_dir + "results.csv", index=False)

    return train_info_dir


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
        "outdir" : "output/",
        "num_actions":2,
        "num_agents": N,
        "state_dim": K,
        "model_name":"logit",
        "corr": corr
    }
    data_dir=generate_dataset_from_logitmodel(dataset_config)

    #----------------------------------

    # model and training configuration for (s,a) sequence data
    model_config = {'seq_len': 16}
    # set architecture type:
    # >STOMP
    if True:
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
        'num_epochs': 1,
        'learning_rate': 5e-5,
        'batch_size': 8,
        'train2test_ratio': 0.8,
        'data_dir': data_dir,
        'data_seed': 0,
        'model_config': model_config,
        'outdir': 'output/',
        'seed': seed
        }

    store_name = train(config)

    print(f'finished. output stored at: {store_name}')
