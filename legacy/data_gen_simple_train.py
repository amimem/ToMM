from data_gen import generate_data
from train import train
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='data generation parameters')
parser.add_argument('--T', type=int,
                    default=1000, help='episode length')
parser.add_argument('--corr', type=float,
                    default=1.0, help='action correlation')
parser.add_argument('--A', type=int,
                    default=2, help='number of actions')
parser.add_argument('--N', type=int,
                    default=10, help='number of ground agents')
parser.add_argument('--M', type=int,
                    default=2, help='number of abstract agents')
parser.add_argument('--K', type=int,
                    default=3, help='state space dimension')
parser.add_argument('--num_seeds', type=int,
                    default=10, help='number of seeds')
parser.add_argument('--action_selection_method', type=str,
                    default='greedy', help='action selection method')
parser.add_argument('--ensemble', type=str,
                    default='sum', help='ensemble method (sum or mix)')
parser.add_argument('--ground_model_name', type=str,
                    default='bitpop', help='ground model name or hashes for loaded model')
parser.add_argument('--output', type=str,
                    default='output/', help='output directory')
args = parser.parse_args()

#default args
datagen_args=vars(args)
train_args = {}
train_args['model_name']='stomp'
train_args['P']=1e6
train_args['M']=2
train_args['L']=100
train_args['n_hidden_layers']=2
train_args['n_features']=2
train_args['num_codebooks']=10
train_args['enc2dec_ratio']=1
train_args['epochs']=20
train_args['learning_rate']=5e-5
train_args['batch_size']=8
train_args['outdir']='output/'
train_args['data_dir']=''
train_args['seed']=0
train_args['data_seed']=0
train_args['use_lr_scheduler']=False
train_args['step_LR']=30
train_args['gamma']=0.1
train_args['checkpoint_interval']=100
train_args['wandb_entity_name']=None
train_args['wandb_group_name']=None
train_args['wandb_job_type_name']=None


if __name__ == '__main__':

    # variable parameters
    corrvec=[0,1] # [0, 0.5, 1.0]   # agent pairwise action correlation
    Nvec=[10,100] # [1e1,1e2,1e3,1e4]   # number of agents
    # Pvec=[1e5,1e6] # [1e4,1e5,1e6,1e7]  # number of parameters

    # datagen vars
    K=8             # state space dimension
    M_sys=1         # number of agent groups
    T=int(1e4)      # number of samples to learn from
    data_seed = 0   # seed of data generation
    single_agent_capcity = 256*100
    # train vars
    M_train=1       # assumed number of agent groups
    epochs=1

    datagen_args['K']=K
    datagen_args['M']=M_sys
    datagen_args['T']=T
    datagen_args['ground_model_name'] = 'bitpop'


    # generate data from bitpop
    hashtype='bitpop_data'
    hash_data_list=[]
    for corr in corrvec:
        datagen_args['corr']=corr
        for N in Nvec:
            datagen_args['N']=N
            bitpop_data_hash=generate_data(datagen_args.copy())
            hash_data_list.append((corr,N,-1,hashtype,bitpop_data_hash))
    df=pd.DataFrame(hash_data_list,columns=['corr','N','P','hashtype','hash'])
    train_args['model_name']='single'
    train_args['M']=M_train
    train_args['data_seed']=data_seed
    train_args['epochs']=epochs


    # train simple
    hashtype='train_simple'
    hash_data_list=[]
    for corr in corrvec:
        for N in Nvec:
            train_args['data_dir']='data_' + df.loc[
                (df['corr']==corr) & (df['N']==N) & (df['hashtype']=='bitpop_data'),'hash'].values[0]
            for P in Pvec:
                train_args['P']=N*single_agent_capacity
                train_simple_hash=train(train_args.copy())
                hash_data_list.append((corr,N,P,hashtype,bitpop_data_hash))
    dftmp=pd.DataFrame(hash_data_list,columns=['corr','N','P','hashtype','hash'])
    df=pd.concat((df,dftmp))


    # generate data from trained simple
    hashtype = 'simple_data'
    hash_data_list=[]
    for corr in corrvec:
        for N in Nvec:
            data_hash=df.loc[
                (df['corr']==corr) & (df['N']==N) & (df['hashtype']=='bitpop_data'),'hash'].values[0]
            for P in Pvec:
                train_hash=df.loc[
                    (df['corr']==corr) & (df['N']==N) & (df['hashtype']=='train_simple') & (df['P']==P),'hash'].values
                datagen_args['ground_model_name'] = data_hash+'/'+train_hash
                simple_data_hash=generate_data(datagen_args.copy())
                hash_data_list.append((corr,N,P,hashtype,bitpop_data_hash))
    dftmp=pd.DataFrame(hash_data_list,columns=['corr','N','P','hashtype','hash'])
    df=pd.concat((df,dftmp))


    # store
    data_store={}
    data_store['datagen_args']=datagen_args
    data_store['train_args']=train_args
    data_store['hashes']=df
    data_filename = f"hashlist_K_{K}_Msys_{M_sys}_T_{T}_Mtrain_{M_train}_Ep_{epochs}_dataseed_{data_seed}"
    np.save(data_filename+".npy",data_store)
    df.to_csv(data_filename, index=False)












    # #train match
    # train_args['model_name']='match'
    # match_train_hashes = []
    # for data_hash in simple_data_hashes:
    #   train_args['data_dir']='data_'+data_hash
    #   match_train_hashes.append(train(train_args.copy()))
    # # write_hashes(match_train_hashes,'match_train')
    # hashlist_dict['match_train']=match_train_hashes
    # print(''.join(['\n']*10))
    # np.save(hashlist_filename,hashlist_dict)

    # #train mlp
    # train_args['model_name']='match'
    # match_train_hashes = []
    # for data_hash in simple_data_hashes:
    #   train_args['data_dir']='data_'+data_hash
    #   match_train_hashes.append(train(train_args.copy()))
    # # write_hashes(match_train_hashes,'match_train')
    # hashlist_dict['match_train']=match_train_hashes
    # print(''.join(['\n']*10))
    # np.save(hashlist_filename,hashlist_dict)

    # #generate data from trained match
    # match_data_hashes=[]
    # for datahash,trainhash in zip(simple_data_hashes,match_train_hashes):
    #   datagen_args['ground_model_name'] = datahash+'/'+trainhash
    #   match_data_hashes.append(generate_data(datagen_args.copy()))
    # write_hashes(match_data_hashes,'match_data')
    # hashlist_dict['match_data']=match_data_hashes
    # print(''.join(['\n']*10))

    # np.save(hashlist_filename,hashlist_dict)


    # def write_hashes(hash_list,hash_name,file_name=hashlist_filename):
    #   with open(file_name,'a') as f:
    #       f.write(hash_name)
    #       for ha in hash_list:
    #           f.write(ha)
    # write_hashes(bitpop_data_hashes,'bitpop_data')
    # write_hashes(simple_train_hashes,'simple_train')
    # write_hashes(simple_data_hashes,'single_data')

