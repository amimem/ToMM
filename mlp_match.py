import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import wandb
import time
import hashlib
import os
import yaml
from types import SimpleNamespace

# import custom functions
from models import STOMP, MLPperagent,sharedMLP,MLPallagents
from data_utils import load_data, ContextDataset, generate_dataset_from_logitmodel

# get slurm job array index
try:
    job_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
except:
    job_id = -1
    print("Not running on a cluster")


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
            if model.decoder_type == 'BuffAtt' and batch_step==0:
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

def get_data_and_configs(config):

        data_dir = config['outdir'] + config['data_dir']
        train_data, test_data, data_config = load_data(data_dir,data_seed=config['data_seed'])
        data_config = SimpleNamespace(**data_config['file_attrs'])
        model_config = SimpleNamespace(**config['model_config'])
        config = SimpleNamespace(**config)
        print(f"loaded data: N={data_config.num_agents}, A={data_config.num_actions}, n_samp={data_config.num_train_samples}, c={data_config.corr}")


        # make a directory for saving the results
        save_dir = os.path.join(config.outdir, config.data_dir, 'training_results/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # hash the arguments except for the outdir
        hash_dict = vars(config).copy()
        hash_dict.pop('outdir')

        # make a subdirectory for each run
        hash_var = hashlib.blake2s(str(hash_dict).encode(), digest_size=5).hexdigest()
        train_dir = os.path.join(save_dir, hash_var)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        # initialize wandb
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        wandb_run_name = str(hash_var) + "_" + str(timestamp)
        print("wandb run name: " + wandb_run_name, flush=True)

        return train_data, test_data, data_config, model_config, config, train_dir

def train(config):

    train_dataset, test_dataset, data_config, model_config, config, train_dir = get_data_and_configs(config)
    
    contextualized_train_data = ContextDataset(train_dataset, model_config.seq_len, data_config.num_actions, check_duplicates=True)
    contextualized_test_data = ContextDataset(test_dataset, model_config.seq_len, data_config.num_actions)
    train_dataloader = DataLoader(
        contextualized_train_data, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        contextualized_test_data, batch_size=config.batch_size, shuffle=False)
    
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
    print(f"number of parameters: {num_parameters} (rel. est. error: {(config.P - num_parameters)/num_parameters:.4f})", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    print(f"seed {config.seed} training of {model_config.model_name} model " +\
        f"with modelsize {model_config.P} for {config.num_epochs} epochs " +\
        f"using batchsize {config.batch_size} and LR {config.learning_rate}", flush=True)

    # wandb.init(project="ToMM", group="archcompare",job_type=None, config=vars(config))
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
            f"took {int(time.time()-st)} s"
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
    with open(train_dir + "/model_config.yaml", 'w') as file:
        yaml.dump(model_config, file)

    torch.save(model.state_dict(), train_dir + "/state_dict_final.pt")

    df.to_csv(train_dir + "results.csv", index=False)

    return train_dir

def get_context_distinguishability_data(config):

    train_dataset, test_dataset, data_config, model_config, config, train_dir = get_data_and_configs(config)
    
    contextualized_train_data = ContextDataset(train_dataset, model_config.seq_len, data_config.num_actions, check_duplicates=True)
    return contextualized_train_data.number_of_contexts_with_duplicates

def assign_parameters():

    #----------------------------------
    set_seed(seed)
    #----------------------------------

    #generate correlation-controlled (s,a)-tuple data
    dataset_config = {
        "num_train_samples": training_sample_budget,
        "num_test_samples": evaluation_sample_size,
        "outdir" : "output/",
        "num_actions":A,
        "num_agents": N,
        "state_dim": S,
        "model_name":"logit",
        "corr": corr
    }

    data_dir=generate_dataset_from_logitmodel(dataset_config)

    #----------------------------------

    # training configuration for (s,a) block data
    train_config = {
        'P': P,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'data_dir': data_dir,
        'data_seed': data_seed,
        'outdir': 'output/',
        'seed': seed
        }

    # add model architexture configuration    
    model_config = {'seq_len': seq_len}
    if False:
        # >STOMP
        model_config['model_name'] = 'STOMP'
        # config['enc_MLPhidden_dim'] = 256
        # config['enc_hidden_dim'] = 256
        # config['enc_out_dim'] = 256
        if False: # --single-agent baseline
            model_config['cross_talk'] = False
            model_config['decoder_type'] = 'MLP'
        else: # --multi-agent baseline
            model_config['cross_talk'] = True
            model_config['decoder_type'] = 'BuffAtt'
            # config['dec_hidden_dim'] = 256
    else:
        # >illustrative baselines
        # config['enc_out_dim'] = 256
        if False: # unshared
            model_config['model_name'] = 'MLPperagent'
        elif True: # shared
            model_config['model_name'] = 'sharedMLP'
        elif False: # 1 network 
            model_config['model_name'] = 'MLPallagents'
        model_config['cross_talk'] = None
        model_config['decoder_type'] = None
    train_config['model_config'] = model_config

    return dataset_config,train_config


if __name__ == "__main__":

    # experiment parameters
    N = 10 # num agents. [10,100,1000]
    corr = 0 # pairwise correlation in data generated from logit model. [0, 0.5, .99]
    P = int(5e5) # training model size. big enough such that hidden width not too small? 
    seq_len = 8 # context length. adjust based on distinguishability of contexts
    training_sample_budget = int(1e4)

    # fixed training data properties
    S = 8 # state space dim 8 gives 2^8=256 distinct observations in ground system policy, big enough even for largest N?
    A = 2 # single-agent action space dim

    # training parameters (set as needed)
    num_epochs = 50
    learning_rate = 5e-5
    batch_size = 8
    data_seed = 0
    seed = 0
    evaluation_sample_size = int(1e4) # large enough for low variability of test accuracy across data_seeds

    dataset_config,train_config=assign_parameters()
    
    if False:
        Nvec=[10,100]
        svec=[4,8,12,16,20]
        corrvec=[0,0.3,0.6,0.9,1.0]
        count_data=[]

        for nit,N in enumerate(Nvec):
            for sit,seq_len in enumerate(svec):
                for cit,corr in enumerate(corrvec):
                    dataset_config['num_agents']=N
                    dataset_config['corr']=corr
                    data_dir=generate_dataset_from_logitmodel(dataset_config)
                    train_config['data_dir'] =data_dir
                    train_config['model_config']['seq_len'] = seq_len
                    count_data.append([N,seq_len,corr,get_context_distinguishability_data(train_config)/training_sample_budget])
        df=pd.DataFrame(count_data,columns=['N','seqlen','corr','count'])
        file_name = 'distinguishability_df.csv'
        df.to_csv(file_name,index=False)
    else:
        store_name = train(train_config)
        print(f'finished. output stored at: {store_name}')
