import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os

from Environment import Environment
from GroundModelJointPolicy import GroundModelJointPolicy

def generate_system_data(sys_parameters, sim_parameters, output_path):
	
	# assign system parameters
	K = sys_parameters['K']   # state space dimensionality
	L = sys_parameters['L']   # abstract actions
	M = sys_parameters['M']   # abstract agents
	N = sys_parameters['N']   # agents
	epsiode_length = sys_parameters['episode_length']
	baseline_name = sys_parameters['baseline_name']
	baseline_paras=sys_parameters['baseline_paras']
	exploit_mode = sys_parameters['exploit_mode']

    #assign data generation parameters
	seedlist = range(sim_parameters['num_seeds'])
    num_steps = sim_parameters['num_episodes']*epsiode_length


	assert int(N/M)==N/M, "number of abstract agents should divide ground agents for some groundmodels"

	# Ground system
	num_agents = N
	state_space_dim = K  # vector space dimensionality
	action_space_dim = 2 # number of discrete actions; fixed to 2 for simplicity
	T = epsiode_length

	# Initialize environment
	env = Environment(K, N, T, 1)

	# Initialize ground system model
	baseline_paras={}
	baseline_name='bitpop'
	if baseline_name == 'bitpop':
	    baseline_paras["corr"]=0.6
	    # baseline_paras['gen_type'] = 'mix'
	    baseline_paras['gen_type'] = 'sum'
	    num_abs_agents = M

	agents_per_abstract_agent=int(N/M)
	groundmodel = GroundModelJointPolicy(
	    K, 
	    M, 
	    agents_per_abstract_agent, 
	    action_space_dim=2, 
	    baseline = baseline_name, 
	    baseline_paras=baseline_paras
	    )
    model_name = 'groundmodel'
    model = groundmodel

    
    #run rollouts
    time_store = []
    state_store = []
    jointaction_store = []
    for seed in seedlist:
	    episode_time_indices = []
	    state_seq = []
	    joint_action_seq = []
	    env.state=env.sample_initial_state(state_space_dim,seed)
	    for step in range(num_steps):

	        state_seq.append(env.state.detach().cpu().numpy())
	        action_probability_vectors = model.forward(env.state)
	        
	        if exploit_mode: #take greedy action
	            actions = torch.argmax(action_probability_vectors,dim=-1)
	        else: #sample
	            actions=Categorial(action_probability_vectors) #Categorical has batch functionality!
	        joint_action_seq.append(actions.detach().cpu().numpy())
	        env.state, episode_step = env.step(env.state, actions)
	        episode_time_indices.append(episode_step)
	    time_store.append(episode_time_indices)
	    state_store.append(state_seq)
	    jointaction_store.append(joint_action_seq)

	#add data to simulation parameter dictionary and store
    sim_parameters['sys_parameters'] = sys_parameters
    sim_parameters["times"] = time_store
    sim_parameters["states"] = state_store
    sim_parameters["actions"] = jointaction_store
    filename=f"{output_path}_trainingdata_{model_name}_exploit{exploit_mode}_{K}_L{L}_M{M}_N{N}_T{epsiode_length}.npy"
    np.save(filename,sim_parameters)

