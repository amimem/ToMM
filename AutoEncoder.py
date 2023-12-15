import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils import get_gumbel_softmax_sample, get_linear_nonlinear_function

class Encoder(nn.Module):
    def __init__(self, state_space_dim, abs_action_space_dim, enc_hidden_dim, num_abs_agents):
        super(Encoder, self).__init__()
        self.state_space_dim = state_space_dim
        self.abs_action_space_dim = abs_action_space_dim
        self.hidden_dim = enc_hidden_dim
        self.num_abs_agents = num_abs_agents
        self.joint_abs_action_dim=abs_action_space_dim*num_abs_agents

        # Define the neural network architecture, E.g.
        self.fc1 = nn.Linear(state_space_dim, enc_hidden_dim)
        self.fc2 = nn.Linear(enc_hidden_dim, self.joint_abs_action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)

        # do gumbel softmax for each abs agent and collect results
        # gumbel_softmax_index_samples = torch.empty(num_abs_agents, abs_action_space_dim, dtype=torch.long)
        abstract_actions = []
        for abs_agent_id in range(self.num_abs_agents):
            abs_agent_logits = logits[abs_agent_id*self.abs_action_space_dim:(abs_agent_id+1)*self.abs_action_space_dim]
            one_hot_vector = get_gumbel_softmax_sample(abs_agent_logits)
            abstract_actions.append(torch.where(one_hot_vector)[0]) #could store as onehot:
            # gumbel_softmax_index_samples[abs_agent_id] = one_hot_vector 
        return abstract_actions 

class Decoder(nn.Module):
    def __init__(self, num_abs_agents,num_agents,abs_action_space_dim, action_space_dim = 2, index_mode = True):
        super(Decoder, self).__init__()
        self.num_agents = num_agents
        #initialize partition
        self.index_mode = index_mode
        if index_mode:
            w = torch.empty(num_agents,num_abs_agents)
            self.partition_logits = nn.init.normal_(w)
            # =self.init_index_mode_partition(num_abs_agents,num_agents) # a matrix
        else: # embedding mode
            embedding_dim = 256
            self.embedded_agents=nn.Embedding(num_embeddings=num_agents, embedding_dim=embedding_dim)
            self.partition_logits= None #? # a differentiable state-to-partition-element function (partition element represented as onehot) that partitions the embedding space into num_abs_agents regions (e.g. if using hyperplanes, would need log2(num_abs_agents) of them)

        #initialize policies (input dimension is 1 (integer abstract action) 1+who it is for index mode and 1+embed_dim for embeding mode)
        input_dim = 1 + 1 if self.index_mode else 1 + embedding_dim
        linear_layer = nn.Linear(input_dim, action_space_dim)
        self.shared_action_policy_logits = nn.Linear(input_dim, action_space_dim)

    # def init_index_mode_partition(num_abs_agents,num_agents, init_prob=0.99):
    #     abs_agent_idxs = np.random.randint(num_abs_agents,size=num_agents)
    #     partition=(1 - init_prob)/(num_abs_agents - 1)*np.ones((num_agents,num_abs_agents)) #so that rows sum to 1
    #     partition[:, abs_agent_idxs] = init_prob
    #     return torch.Tensor(partition)

    def get_abs_agent_assignment_logits(self,agent_idx,index_mode=True):
        if index_mode:
            return self.partition_logits[agent_idx]
        else:
            return self.partition_logits(agent_idx)

    def get_abs_agent(self, agent_idx): # can this be vectorized to avoid the loop over all agents?
        abs_agent_assignment_logits = self.get_abs_agent_assignment_logits(agent_idx)
        one_hot_vector=get_gumbel_softmax_sample(abs_agent_assignment_logits)
        abs_agent_idx = int(torch.where(one_hot_vector)[0][0])
        return abs_agent_idx

    def forward(self, abs_actions):
        actions=[]
        for agent_idx in range(self.num_agents):
            # given the partition, get it's corresponding abstract agent info 
            abs_agent_idx = self.get_abs_agent(agent_idx)
            print(abs_agent_idx)
            if self.index_mode:
                input_tensor = torch.FloatTensor([abs_actions[abs_agent_idx],agent_idx])
            else:
                input_tensor = torch.FloatTensor([abs_actions[abs_agent_idx],self.embedded_agents[agent_idx]])
            action_probability_vector = F.softmax(self.shared_action_policy_logits(input_tensor))
            
            #take greedy action
            action = torch.argmax(action_probability_vector)  #really this is part of the policy
            
            actions.append(action)
        return actions
