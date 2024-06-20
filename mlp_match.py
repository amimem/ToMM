import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import wandb
import time
import hashlib
import argparse
import os
import yaml
from types import SimpleNamespace

# import custom functions
from models import STOMP, MLPbaselines
from data_utils import load_data, gen_logit_dataset, ContextDataset

# Create the parser
parser = argparse.ArgumentParser(description='Experiment parameters')

# Add arguments
parser.add_argument('--N', type=int, default=10, help='num agents. [10,100,1000]')
parser.add_argument('--corr', type=float, default=0, help='pairwise correlation in data generated from logit model. [0, 0.5, .99]')
parser.add_argument('--P', type=int, default=int(5e5), help='training model size.')
parser.add_argument('--seq_len', type=int, default=8, help='context length.')
parser.add_argument('--training_sample_budget', type=int, default=int(1e4), help='training sample budget')

# Fixed training data properties
parser.add_argument('--S', type=int, default=8, help='state space dimension')
parser.add_argument('--A', type=int, default=2, help='single-agent action space dimension')

# Training parameters
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--data_seed', type=int, default=0, help='data seed for generating training data')
parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')
parser.add_argument('--evaluation_sample_size', type=int, default=int(1e4), help='sample size for evaluation')

# Parse the arguments
args = parser.parse_args()


# get slurm job array index
try:
    job_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
except:
    job_id = -1
    print("Not running on a cluster")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# sets the seed for generating random numbers
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# iterates over dataset, evaluating model response and optionally updating parameters
def eval_performance(model, dataloader, and_train=False, optimizer=None):
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
        action_logit_vectors = model(state_seq.to(device), action_seqs.to(device))

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
        
        model_config.num_actions = data_config.num_actions
        model_config.num_agents = data_config.num_agents
        model_config.state_dim = data_config.state_dim
        model_config.P = config.P

        # make a directory for saving the results
        save_dir = os.path.join(config.outdir, config.data_dir, 'training_results/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # hash the arguments except for the outdir
        hash_dict = vars(config).copy()
        hash_dict.pop('outdir')

        # make a subdirectory for each run
        hash_var = hashlib.blake2s(str(hash_dict).encode(), digest_size=5).hexdigest()
        config.hash=hash_var
        train_dir = os.path.join(save_dir, hash_var)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        config.hash = hash_var

        run_dict = {
            "corr": data_config.corr,
            "state_dim": data_config.state_dim,
            "num_actions": model_config.num_actions,
            "num_train_samples":data_config.num_train_samples,
            "num_test_samples":data_config.num_test_samples,
            "num_agents": data_config.num_agents,
            "data_seed": config.data_seed,
            "data_hash": config.data_dir,
            "model_name": model_config.model_name,
            "decoder_type": model_config.decoder_type,
            "P": model_config.P,
            "decoder_type": model_config.decoder_type,
            "seq_len": model_config.seq_len,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size
        }

        return train_data, test_data, data_config, model_config, config, run_dict, train_dir

def train(config):

    train_dataset, test_dataset, data_config, model_config, config, run_dict, train_dir = get_data_and_configs(config)
    
    contextualized_train_data = ContextDataset(train_dataset, model_config.seq_len, data_config.num_actions)
    contextualized_test_data = ContextDataset(test_dataset, model_config.seq_len, data_config.num_actions)
    train_dataloader = DataLoader(
        contextualized_train_data, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        contextualized_test_data, batch_size=config.batch_size, shuffle=False)
 
    if model_config.model_name=='STOMP':
        model = STOMP(model_config).to(device)
    elif model_config.model_name.split('_')[0]=='MLP':
        model = MLPbaselines(model_config).to(device)

    # log number of parameters
    model_config.Pactual = model.count_parameters()
    run_dict['Pactual'] = model_config.Pactual
    print(f"number of parameters: {model_config.Pactual} (rel. est. error: {(config.P - model_config.Pactual)/model_config.Pactual:.4f})", flush=True)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    print(f"seed {config.seed} training of {model_config.model_name} model " +\
        f"with modelsize {model_config.Pactual} for {config.num_epochs} epochs " +\
        f"using batchsize {config.batch_size} and LR {config.learning_rate}", flush=True)

    # initialize wandb
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    wandb_run_name = str(config.hash) + "_" + str(timestamp)
    print("wandb run name: " + wandb_run_name, flush=True)
    run=wandb.init(project="ToMMM", group="archcompare",job_type=None, config=run_dict)
    locally_logged =[]
    for epoch in range(config.num_epochs):
        st=time.time()
        train_epoch_loss, train_epoch_accuracy = eval_performance(
            model, train_dataloader, and_train=True, optimizer=optimizer)
        test_epoch_loss, test_epoch_accuracy = eval_performance(
            model, test_dataloader)

        print(
            f"Epoch {epoch+1}/{config.num_epochs}, " +
            f"Train/Test Loss: {train_epoch_loss:.4f}/{test_epoch_loss:.4f}, " +
            f"Accuracy: {train_epoch_accuracy:.4f}/{test_epoch_accuracy:.4f}, " +
            f"took {int(time.time()-st)} s"
        )
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_epoch_loss,
            "train_accuracy": train_epoch_accuracy,
            "test_loss": test_epoch_loss,
            "test_accuracy": test_epoch_accuracy
        }
        run.log(epoch_data)
        locally_logged.append(epoch_data.update(run_dict))

    df = pd.DataFrame(locally_logged)
    wandb.finish()

    training_run_info = \
        f"_{model_config.model_name}"+\
        f"_modelsize_{model_config.P}"+\
        f"_trainseed_{config.seed}"+\
        f"_epochs_{config.num_epochs}"+\
        f"_batchsize_{config.batch_size}"+\
        f"_lr_{config.learning_rate}"
    print("saving " + training_run_info + " at hash:", flush=True)
    print(config.hash, flush=True)

    # also save config as yaml
    with open(train_dir + "/train_config.yaml", 'w') as file:
        yaml.dump(config, file)

    torch.save(model.state_dict(), train_dir + "/state_dict_final.pt")

    df.to_csv(train_dir + "/results.csv", index=False)

    return train_dir

def get_context_distinguishability_data(config):

    train_dataset, test_dataset, data_config, model_config, config, run_config, train_dir = get_data_and_configs(config)
    
    contextualized_train_data = ContextDataset(train_dataset, model_config.seq_len, data_config.num_actions, check_duplicates=True)
    return contextualized_train_data.number_of_contexts_with_duplicates

def collect_parameters_and_gen_data():

    #----------------------------------
    set_seed(args.seed)
    #----------------------------------

    #generate correlation-controlled (s,a)-tuple data
    dataset_config = {
        "num_train_samples": args.training_sample_budget,
        "num_test_samples": args.evaluation_sample_size,
        "outdir" : "output/",
        "num_actions": args.A,
        "num_agents": args.N,
        "state_dim": args.S,
        "model_name":"logit",
        "corr": args.corr
    }

    data_dir=gen_logit_dataset(dataset_config)

    #----------------------------------
    # training configuration for (s,a) block data
    train_config = {
        'P': args.P,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'data_dir': data_dir,
        'data_seed': args.data_seed,
        'outdir': 'output/',
        'seed': args.seed
        }

    # add model architexture configuration    
    model_config = {'seq_len': args.seq_len}
    if True:
        # >STOMP
        model_config['model_name'] = 'STOMP'
        # config['enc_MLPhidden_dim'] = 256
        # config['enc_hidden_dim'] = 256
        # config['enc_out_dim'] = 256
        if True: # --single-agent baseline
            model_config['cross_talk'] = False
            model_config['decoder_type'] = 'MLP'
        else: # --multi-agent baseline
            model_config['cross_talk'] = False
            model_config['decoder_type'] = 'BuffAtt'
            # config['dec_hidden_dim'] = 256
    '''
        # >illustrative baselines
        # config['enc_out_dim'] = 256
        if False: # unshared
            model_config['model_name'] = 'MLP_nosharing'
        elif True: # shared
            model_config['model_name'] = 'MLP_fullsharing' 
        elif False: # 1 network 
            model_config['model_name'] = 'MLP_encodersharingonly'
        model_config['cross_talk'] = None
        model_config['decoder_type'] = None
    '''

    train_config['model_config'] = model_config

    return dataset_config,train_config


if __name__ == "__main__":

    #----------------------------------
    dataset_config,train_config=collect_parameters_and_gen_data()
    
    """
    # experiment parameters
    N = 10 # num agents. [10,100,1000]
    corr = 0.8 # pairwise correlation in data generated from logit model. [0, 0.5, .99]
    P = int(5e5) # training model size. big enough such that hidden width not too small? 
    seq_len = 16 # context length. adjust based on distinguishability of contexts
    training_sample_budget = int(1e4)

    # fixed training data properties
    S = 8 # state space dim 8 gives 2^8=256 distinct observations in ground system policy, big enough even for largest N?
    A = 2 # single-agent action space dim

    # training parameters (set as needed)
    num_epochs = 50
    learning_rate = 5e-4
    batch_size = 8
    data_seed = 0
    seed = 0
    evaluation_sample_size = int(1e4) # large enough for low variability of test accuracy across data_seeds

    dataset_config,train_config=collect_parameters_and_gen_data()
    
    #----------------------------------
    set_seed(seed) # 
    #----------------------------------

    if False: #distinguishability analysis

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
    """

    #training experiment
    store_name = train(train_config)
    print(f'finished. output stored at: {store_name}')
