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
        logit_array = torch.reshape(logits,(self.abs_action_space_dim,self.num_abs_agents))
        one_hot_array = get_gumbel_softmax_sample(logit_array)
        abstract_actions = torch.argmax(one_hot_array, dim=1)
        return abstract_actions

class Decoder(nn.Module): #TODO: pullout assigner code into its own class
    def __init__(self, num_abs_agents,num_agents,abs_action_space_dim, action_space_dim = 2, index_mode = False):
        super(Decoder, self).__init__()
        self.num_agents = int(num_agents)
        self.index_mode = index_mode

        #initialize assigner
        w = torch.empty(num_agents,num_abs_agents)
        self.assigner_logit_array = nn.init.normal_(w)
        
        #initialize ground agent embedding
        embedding_dim = 256
        self.agent_embedding=nn.Embedding(num_embeddings=num_agents, embedding_dim=embedding_dim)

        #initialize policies (input dimension is 1 (integer abstract action) 1+who it is for index mode and 1+embed_dim for embeding mode)
        input_dim = 1 + 1 if self.index_mode else 1 + embedding_dim
        linear_layer = nn.Linear(input_dim, action_space_dim)
        self.shared_action_policy_network = nn.Linear(input_dim, action_space_dim)

    def forward(self, abs_actions):

        one_hot_assignment_array = get_gumbel_softmax_sample(self.assigner_logit_array)
        abstract_agent_assignments = torch.argmax(one_hot_assignment_array, dim=1)
        assigned_abstract_actions = abs_actions[abstract_agent_assignments]
        if self.index_mode:
            input_tensor = torch.cat([assigned_abstract_actions[:,None],torch.range(num_agents)[:,None]],dim=-1)
        else:
            input_tensor = torch.cat([assigned_abstract_actions[:,None],self.agent_embedding(torch.LongTensor(range(self.num_agents)))],dim=-1)
        action_probability_vectors = F.softmax(self.shared_action_policy_network(input_tensor))

        return action_probability_vectors