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
import copy
from types import SimpleNamespace
from functools import partial
from scipy.optimize import minimize_scalar
# import custom functions
from models import STOMP, MLPbaselines
from data_utils import load_data, get_logit_dataset_pathname, ContextDataset

# Create the parser
parser = argparse.ArgumentParser(description='Experiment parameters')

# Add arguments
parser.add_argument('--N', type=int, default=10, help='num agents. [10,100,1000]')
parser.add_argument('--corr', type=float, default=0.8, help='pairwise correlation in data generated from logit model. [0, 0.5, .99]')
parser.add_argument('--P', type=int, default=int(5e5), help='training model size.')
parser.add_argument('--seq_len', type=int, default=16, help='context length.')
parser.add_argument('--training_sample_budget', type=int, default=int(1e4), help='training sample budget')
parser.add_argument('--use_pos_enc', type=int, default=1, help='if 1, use positional encodings, else do not')
parser.add_argument('--inter', type=str, default='None', help='label of interaction model to use (None,ISAB,attn,ipattn, ...)')


# Fixed training data properties
parser.add_argument('--S', type=int, default=8, help='state space dimension')
parser.add_argument('--A', type=int, default=2, help='single-agent action space dimension')
parser.add_argument('--wagent', type=float, default=1.0, help='weight of agent-dependence')
parser.add_argument('--state_corr_len', type=float, default=1.0, help='state correlation length')

# Training parameters
parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--data_seed', type=int, default=0, help='data seed for generating training data')
parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')
parser.add_argument('--evaluation_sample_size', type=int, default=int(1e4), help='sample size for evaluation')
parser.add_argument('--use_wandb', type=int, default=1, help='if 1, log results using wandb, else do not')

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
def eval_performance(model, dataloader, and_train=False, optimizer=None, criterion=None, agent_subset_fraction=0.1):
    if and_train:
        model.train()
    else:
        model.eval()

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
                action_logit_vectors[:, agent_idx, :], target_actions[:, agent_idx].to(device))
            for agent_idx in range(dataloader.dataset.num_agents)
        )

        if and_train:
            loss.backward()
            optimizer.step()

        accuracy = (torch.argmax(action_logit_vectors, dim=-1)
                    == target_actions.to(device)).float().mean().item()
        batch_loss.append(loss.item())
        batch_accuracy.append(accuracy)

    return np.mean(batch_loss), np.mean(batch_accuracy)

def get_data_and_configs(config):

    data_dir = config['outdir'] + config['data_dir']
    train_data, test_data, data_config = load_data(data_dir,data_seed=config['data_seed'])
    data_config = SimpleNamespace(**data_config['file_attrs'])
    model_config = SimpleNamespace(**config['model_settings'])
    config = SimpleNamespace(**config)
    print(f"loaded data: N={data_config.num_agents}, A={data_config.num_actions}, n_samp={data_config.num_train_samples}, c={data_config.corr}, wagent={data_config.agent_weight}")
    
    model_config.num_actions = data_config.num_actions
    model_config.num_agents = data_config.num_agents
    model_config.state_dim = data_config.state_dim
    model_config.P = config.P
    model_config.use_pos_enc = config.use_pos_enc

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
        "wagent": data_config.agent_weight,
        "state_corr_len": data_config.state_corr_len,
        "state_dim": data_config.state_dim,
        "num_actions": model_config.num_actions,
        "num_train_samples":data_config.num_train_samples,
        "num_test_samples":data_config.num_test_samples,
        "num_agents": data_config.num_agents,
        "data_seed": config.data_seed,
        "data_hash": config.data_dir,
        "hash": config.hash,
        "model_name": model_config.model_name,
        "decoder_type": model_config.decoder_type,
        "P": model_config.P,
        "inter_model_type": model_config.inter_model_type,
        "use_pos_enc": model_config.use_pos_enc,
        "seq_len": model_config.seq_len,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size
    }

    return train_data, test_data, data_config, model_config, config, run_dict, train_dir

def get_grad_norms(model, model_config):
    #store grad norms of each module
    module_name_list = ['seq_enc.fc_in', 'seq_enc.LSTM', 'seq_enc.attn', 'decoder.model']
    module_gradnorms = {name:[] for name in module_name_list}
    all_gradnorms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad.detach().flatten().cpu()
            all_gradnorms[name]=torch.linalg.vector_norm(grad).numpy()/len(grad)
            for mit, module_name in enumerate(module_name_list):
                if module_name in name:
                    module_gradnorms[module_name].append(all_gradnorms[name])
    for mit, module_name in enumerate(module_name_list):
        module_gradnorms[module_name] = np.mean(module_gradnorms[module_name])
    return module_gradnorms, all_gradnorms

def get_losschange_over_batch(learning_rate, model, orig_paras, dataloader, criterion, device,num_test_batches):
    delta_loss_batch = []
    model.load_state_dict(orig_paras)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for batch_step, (state_seq, action_seqs, target_actions) in enumerate(dataloader):
        action_logit_vectors = model(state_seq.to(device), action_seqs.to(device))
        loss = sum(
            criterion(
                action_logit_vectors[:, agent_idx, :], target_actions[:, agent_idx].to(device)
            ) for agent_idx in range(dataloader.dataset.num_agents)
        )
        pre_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        action_logit_vectors = model(state_seq.to(device), action_seqs.to(device))
        loss = sum(
            criterion(
                action_logit_vectors[:, agent_idx, :], target_actions[:, agent_idx].to(device)
            ) for agent_idx in range(dataloader.dataset.num_agents)
        )
        post_loss = loss.item()
        delta_loss_batch.append(post_loss - pre_loss)

        if batch_step > num_test_batches:
            break

    return sum(delta_loss_batch)/num_test_batches

def find_lr(optimizer, model, dataloader, criterion, device=device):
    # find learning rate
    min_lr,max_lr = 0.00001, 0.01
    num_lr_values = 10
    num_test_batches = 100
    learning_rates = np.logspace(np.log10(min_lr),np.log10(max_lr), num_lr_values)
    orig_state_dict = copy.deepcopy(model.state_dict())
    losschange_fn = partial(get_losschange_over_batch, model=model, orig_paras=orig_state_dict, dataloader=dataloader, criterion=criterion, device=device, num_test_batches=num_test_batches)

    # grid search
    delta_loss = [losschange_fn(learning_rate) for learning_rate in learning_rates]
    print(delta_loss)
    opt_ind =np.argmin(delta_loss)
    opt_learning_rate = learning_rates[opt_ind]
    print(f"Is {opt_learning_rate:.5f} in interior of {min_lr:.5f} and {max_lr:.5f}?")
    if opt_learning_rate > min_lr and opt_learning_rate < max_lr:
        # polish
        iterations = 10
        low, high = learning_rates[opt_ind-1], learning_rates[opt_ind+1]
        low, high = low if low > min_lr else min_lr, high if high < max_lr else max_lr
        print(f"{low} {high}")
        opt_learning_rate=minimize_scalar(losschange_fn, bracket=(low, high), method='Brent', options={'disp':True, "maxiter": iterations}).x
        print(f"res:{opt_learning_rate}")
    return opt_learning_rate

def train(config):

    train_dataset, test_dataset, data_config, model_config, config, run_dict, train_dir = get_data_and_configs(config)

    # data
    contextualized_train_data = ContextDataset(train_dataset, model_config.seq_len, data_config.num_actions)
    contextualized_test_data = ContextDataset(test_dataset, model_config.seq_len, data_config.num_actions)
    train_dataloader = DataLoader(
        contextualized_train_data, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        contextualized_test_data, batch_size=config.batch_size, shuffle=False)
 
    # model
    if model_config.model_name=='STOMP':
        model = STOMP(model_config, device).to(device)
        #check model device
        print(f"Model's device: {next(model.parameters()).device}")
    elif model_config.model_name.split('_')[0]=='MLP':
        model = MLPbaselines(model_config).to(device)
    else:
        raise ValueError("model name not recognized")
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # loss
    criterion = nn.CrossEntropyLoss()

    # pretraining hyper-parameter tuning
    config.learning_rate = args.learning_rate #find_lr(optimizer, model, train_dataloader, criterion)

    # reinitialize to be safe (should reset seed?).
    model = STOMP(model_config, device).to(device)
    # check model device
    print(f"Model's device: {next(model.parameters()).device}")
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # prine all named parameters
    for name, param in model.named_parameters():
        print(name, param.shape)

    # logging number of parameters
    model_config.Pactual = model.count_parameters()
    run_dict['Pactual'] = model_config.Pactual
    print(f"number of parameters: {model_config.Pactual} (rel. est. error: {(config.P - model_config.Pactual)/model_config.Pactual:.4f})", flush=True)
    print(f"seed {config.seed} training of {model_config.model_name} model " +\
        f"with modelsize {model_config.Pactual} on length-{model_config.seq_len} data for {config.num_epochs} epochs " +\
        f"using batchsize {config.batch_size} and LR {config.learning_rate}", flush=True)

    # initialize wandb
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    if args.use_wandb:
        wandb_run_name = '_'.join([sym+str(run_dict[key]) for sym,key in zip(
            ['N','P','l','c','sc','lr','im','dt','pe','wag'],
            ['num_agents','Pactual','seq_len','corr','state_corr_len','learning_rate','inter_model_type','decoder_type','use_pos_enc','wagent']
            )])
        run=wandb.init(project="ToMMM", entity="abstraction", group="post_aamas",job_type=None, config=run_dict, name=wandb_run_name)


    # train
    locally_logged =[]
    for epoch in range(config.num_epochs):
        st=time.time()
        train_epoch_loss, train_epoch_accuracy = eval_performance(
            model, train_dataloader, and_train=True, optimizer=optimizer, criterion=criterion)
        module_gradnorms, all_gradnorms = get_grad_norms(model, model_config)

        test_epoch_loss, test_epoch_accuracy = eval_performance(
            model, test_dataloader, optimizer=optimizer, criterion=criterion)

        print(
            f"Epoch {epoch+1}/{config.num_epochs}, " +
            f"Train/Test Loss: {train_epoch_loss/data_config.num_agents:.4f}/{test_epoch_loss/data_config.num_agents:.4f}, " +
            f"Accuracy: {train_epoch_accuracy:.4f}/{test_epoch_accuracy:.4f}, " +
            f"took {int(time.time()-st)} s"
        )
        epoch_data = {
            "epoch": epoch,
            "train/loss_per_agent": train_epoch_loss/data_config.num_agents,
            "train/accuracy": train_epoch_accuracy,
            "test/loss_per_agent": test_epoch_loss/data_config.num_agents,
            "test/accuracy": test_epoch_accuracy
        }
        epoch_data.update(module_gradnorms)
        all_gradnorms = dict(zip(['allgrads/'+key for key in all_gradnorms.keys()], all_gradnorms.values())) 
        epoch_data.update(all_gradnorms)
        if args.use_wandb:
            run.log(epoch_data)
        epoch_data.update(run_dict)
        locally_logged.append(epoch_data)

    df = pd.DataFrame(locally_logged)
    if args.use_wandb:
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
        yaml.dump(vars(config), file)
    with open(train_dir + "/model_config.yaml", 'w') as file:
        yaml.dump(vars(model_config), file)

    torch.save(model.state_dict(), train_dir + "/state_dict_final.pt")

    df.to_csv(train_dir + "/results.csv", index=False)

    return train_dir

def get_context_distinguishability_data(config):

    train_dataset, test_dataset, data_config, model_config, config, run_config, train_dir = get_data_and_configs(config)
    
    contextualized_train_data = ContextDataset(train_dataset, model_config.seq_len, data_config.num_actions, check_duplicates=True)
    return contextualized_train_data.number_of_contexts_with_duplicates

def collect_parameters_and_gen_data():

    set_seed(args.seed)

    #generate correlation-controlled (s,a)-tuple data
    dataset_config = {
        "num_train_samples": args.training_sample_budget,
        "num_test_samples": args.evaluation_sample_size,
        "outdir" : "output/",
        "num_actions": args.A,
        "num_agents": args.N,
        "state_dim": args.S,
        "model_name":"logit",
        "corr": args.corr,
        "agent_weight": args.wagent,
        "state_corr_len":args.state_corr_len
    }

    data_dir=get_logit_dataset_pathname(dataset_config)

    # training configuration for (s,a) block data
    train_config = {
        'P': args.P,
        'use_pos_enc': args.use_pos_enc,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'data_dir': data_dir,
        'data_seed': args.data_seed,
        'outdir': 'output/',
        'seed': args.seed
        }

    # add model architexture configuration    
    model_settings = {'seq_len': args.seq_len}
    model_settings['model_name'] = 'STOMP'
    model_settings['inter_model_type'] = args.inter if args.inter!='None' else None #'rnn','attn','ipattn','SAB','ISAB'
    model_settings['decoder_type'] = 'MLP'
    # model_settings['decoder_type'] = 'BuffAtt'

    train_config['model_settings'] = model_settings

    return dataset_config,train_config


if __name__ == "__main__":
    #at N=10
    #args.wagent=[0,1,10]
    #args.use_pos_enc=[0,1]
    #args.seq_len=[1,16]
    #args.inter_model=[None,attn]

    dataset_config,train_config=collect_parameters_and_gen_data()

    store_name = train(train_config)
    print(f'finished. output stored at: {store_name}')
