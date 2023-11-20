import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import logging
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)

def get_gumbelsoftmax_sample(logit_vector, hard_flag=True, tau=1):
    gumbel_dist = distributions.Gumbel(0, 1)
    gumbel_noise = gumbel_dist.sample(logit_vector.shape)
    one_hot_vector = F.gumbel_softmax(logit_vector + gumbel_noise, tau=tau, hard=hard_flag)
    return one_hot_vector

def get_linearnonlinear_function(input_dim, output_dim):
    transition_mtr= nn.Linear(output_dim,input_dim) #should be fixed. need to detach?
    def func(input_tensor):
        return F.sigmoid(torch.matmul(transition_mtr,input_tensor))
    return func

class Encoder(nn.Module):
    def __init__(self, state_dim, abs_action_space_dim, hidden_dim, num_abs_agents):
        super().__init__()
        self.state_dim = state_dim
        self.abs_action_space_dim = abs_action_space_dim
        self.hidden_dim = hidden_dim
        self.joint_abs_action_dim=abs_action_space_dim*num_abs_agents

        # Define the neural network architecture, E.g.
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.joint_abs_action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(torch.Tensor([state])))
        logits = self.fc2(x)

        # do gumbel softmax for each abs agent and collect results
        # gumbel_softmax_index_samples = torch.empty(num_abs_agents, abs_action_space_dim, dtype=torch.long)
        abstract_actions = []
        for abs_agent_id in range(num_abs_agents):
            abs_agent_logits = logits[abs_agent_id*abs_action_space_dim:(abs_agent_id+1)*abs_action_space_dim]
            one_hot_vector = get_gumbelsoftmax_sample(abs_agent_logits)
            abstract_actions.append(torch.where(one_hot_vector)[0]) #could store as onehot:
            # gumbel_softmax_index_samples[abs_agent_id] = one_hot_vector 
        return abstract_actions 

class Decoder(nn.Module):
    def __init__(self, num_abs_agents,num_agents,abs_action_space_dimension, action_space_dim = 2):
        super(Decoder, self).__init_

        #initialize partition
        index_mode = True
        if index_mode:
            self.partition=init_index_mode_partition(num_abs_agents,num_agents) # a matrix
        else: # embedding mode
            embedding_dim = 10
            self.embedded_agents=nn.Embedding(num_embeddings=num_agents, embedding_dim=embedding_dim)
            self.partition= None #? # a differentiable state-to-partition-element function (partition element represented as onehot) that partitions the embedding space into num_abs_agents regions (e.g. if using hyperplanes, would need log2(num_abs_agents) of them)
        
        def get_abs_agent_assignment_probabilities(self,agent_idx,index_mode=True):
            if index_mode:
                return self.partition[agent_idx]
            else:
                return self.partition(agent_idx)

        #initialize policies (input dimension is 2: integer abstract action and integer abstract agent index)
        self.action_policies =[get_linearnonlinear_function(action_space_dim, 2) for i in range(num_agents)]

    def init_index_mode_partition(num_abs_agents,num_agents, init_prob=0.99):
        abs_agent_idxs = np.random.randint(num_abs_agents,size=num_agents)
        partition=(1 - init_prob)/(num_abs_agents - 1)*np.ones((num_agents,num_abs_agents)) #so that rows sum to 1
        partition[:, abs_agent_idxs] = init_prob
        return torch.Tensor(partition)

    def decode(self, abs_actions):

        # for each ground agent
        actions=[]
        for agent_idx, action_policy in enumerate(self.action_policies):
            # given the partition, get it's corresponding abstract agent info 
            abs_agent_idx, abs_action = get_abs_agent_and_action(abs_actions,agent_idx) # can this be vectorized to avoid the loop over all agents?

            # put that through an agent specific network that gives the action
            action_weight = action_policy(torch.FloatTensor([abs_agent_idx, abs_action]))
            action = action_weight > 0 #really this is part of the policy
            actions.append(action)
        return actions

    def get_abs_agent_and_action(abs_actions,agent_idx):
        abs_agent_assignment_probabilities = get_abs_agent_assignment_probabilities(agent_idx)
        logits = torch.log(abs_agent_assignment_probabilities.shape/(1-assignment_probabilities.shape))
        one_hot_vector=get_gumbelsoftmax_sample(logits)
        abs_agent_idx = torch.where(one_hot_vector)[0]
        return abs_agent_idx, abs_actions[abs_agent_idx]

class GumbelPartitionModel(nn.Module):
    def __init__(self, state_space_dim, abs_action_space_dim, hidden_dim, num_abs_agents, action_space_dim=2):
        super().__init__()
        self.encoder = Encoder(
            state_space_dim,
            abs_action_space_dim,
            hidden_dim,
            num_abs_agents
            )
        self.decoder = Decoder(
            abs_action_space_dim,
            action_space_dim)

    def forward(self, state, hard_flag=True):
        abs_actions = self.encoder.forward(state, hard_flag)
        logging.info(f"Abstract actions: {abs_actions}")
        actions = self.decoder.decode(abs_actions)
        return actions


class Environment(state_space_dim,num_agents):
    def __init__(self,):
        state=sample_initial_state(state_space_dim)
        transition_step_func=get_linearnonlinear_function(state_space_dim,state_space_dim+num_agents)

    def sample_initial_state(state_space_dim):
        return torch.Tensor(np.random.rand(state_space_dim)) # is this right?
    
    def step(state,actions):
        return transition_step_func(torch.vstack((self.state,torch.Tensor(actions))))

writer = SummaryWriter()

# System parameters
K = 10   # states
L = 10   # abstract actions
M = 10   # abstract agents
N = 100  # agents

# Ground system
num_agents = N
state_space_dim = K  # vector space dimensionality
action_space_dim = 2 # number of discrete actions; fixed to 2 for simplicity

# Abstracted system
num_abs_agents = M
abs_action_space_dim = L # number of discrete abstract actions
#abstract action policy network parameters
hidden_dim = 256

env = Environment(state_space_dim, num_agents)

model = GumbelPartitionModel(
    state_space_dim,
    abs_action_space_dim,
    hidden_dim,
    num_abs_agents,
    action_space_dim=action_space_dim
    )

if __name__ == '__main__':
    num_steps = 5
    for step in range(num_steps):
        logging.info(f"Step: {step}")
        writer.add_graph(model, env.state)
        writer.close()
        logging.info(f"Input state: {env.state}")
        actions = model.forward(env.state, hard_flag=False)
        logging.info(f"Output actions: {actions}")
        env.state = env.step(env.state,actions)