import torch
from torch import nn
from torch.nn import functional as F
from utils import get_gumbel_softmax_sample, get_linear_nonlinear_function

class AbstractionModelJointPolicy(nn.Module):
    def __init__(self, state_space_dim, abs_action_space_dim, enc_hidden_dim, num_agents, num_abs_agents, action_space_dim=2):
        super(AbstractionModelJointPolicy, self).__init__()
        
        # Define the encoder and decoder
        self.abstract_jointpolicy = Encoder(state_space_dim, abs_action_space_dim, enc_hidden_dim, num_abs_agents)
        self.ground_jointpolicy = Decoder(num_abs_agents, num_agents, abs_action_space_dim, action_space_dim)
        self.assigner = Assigner(num_abs_agents, num_agents)

    def forward(self, state):
        # Pass the state through the encoder to get abstract actions
        abs_actions = self.abstract_jointpolicy(state)
        
        #sample assigner to get assignments
        abstract_agent_assignments = self.assigner()

        # Pass the abstract actions ans assignments through the decoder to get action probabilities
        ground_action_probability_vectors = self.ground_jointpolicy(abs_actions, abstract_agent_assignments)
        
        return ground_action_probability_vectors

class Encoder(nn.Module):
    def __init__(self, state_space_dim, abs_action_space_dim, enc_hidden_dim, num_abs_agents):
        super(Encoder, self).__init__()
        self.state_space_dim = state_space_dim
        self.abs_action_space_dim = abs_action_space_dim
        self.hidden_dim = enc_hidden_dim
        self.num_abs_agents = num_abs_agents
        self.joint_abs_action_dim=abs_action_space_dim*num_abs_agents

        # Define the neural network architecture, E.g.
        independent_mode = False
        if independent_mode:
            enc_hidden_dim_per_abs_agent = enc_hidden_dim/num_abs_agents
            self.fc1 = [nn.Linear(state_space_dim, enc_hidden_dim_per_abs_agent) for idx in range(num_abs_agents)]
            self.fc2 = [nn.Linear(enc_hidden_dim_per_abs_agent, self.joint_abs_action_dim) for idx in range(num_abs_agents)]
        else:
            self.fc1 = nn.Linear(state_space_dim, enc_hidden_dim)
            self.fc2 = nn.Linear(enc_hidden_dim, self.joint_abs_action_dim)

    def forward(self, state, independent_mode=False):
        if independent_mode:
            x = [F.relu(self.fc1[idx](state)) for idx in range(self.num_abs_agents)]
            logits = [self.fc2[idx](xelem) for xlem in x]
            logits = torch.Tensor([item for row in logits for item in row])
        else:
            x = F.relu(self.fc1(state))
            logits = self.fc2(x)
        logit_array = torch.reshape(logits,(self.num_abs_agents,self.abs_action_space_dim))
        one_hot_array = get_gumbel_softmax_sample(logit_array)
        abstract_actions = torch.argmax(one_hot_array, dim=-1)
        return abstract_actions

class Decoder(nn.Module):
    def __init__(self, num_abs_agents,num_agents,abs_action_space_dim, action_space_dim = 2):
        super(Decoder, self).__init__()
        self.num_agents = int(num_agents)

        #initialize ground agent embedding
        embedding_dim = 256
        self.groundagent_embedding=nn.Embedding(num_embeddings=num_agents, embedding_dim=embedding_dim)

        #initialize policies (input dimension is 1 (abstract action index) + embed_dim)
        self.shared_action_policy_network = nn.Linear(1 + embedding_dim, action_space_dim)

    def forward(self, abs_actions, abstract_agent_assignments):
        assigned_abstract_actions = abs_actions[abstract_agent_assignments]
        input_tensor = torch.cat([assigned_abstract_actions[:,None],self.groundagent_embedding(torch.LongTensor(range(self.num_agents)))],dim=-1)
        action_probability_vectors = F.softmax(self.shared_action_policy_network(input_tensor),dim=-1)
        return action_probability_vectors

class Assigner(nn.Module):
    def __init__(self, num_abs_agents,num_agents):
        super(Assigner, self).__init__()
        self.assigner_embedding_dim = num_abs_agents # components are abstract agent weights
        
        #initialize ground agent embedding
        self.abs_agent_assignment_embedding=nn.Embedding(num_embeddings=num_agents, embedding_dim=self.assigner_embedding_dim)
        self.assigner_logit_array = self.abs_agent_assignment_embedding(torch.LongTensor(range(num_agents)))
    
    def forward(self):
        one_hot_assignment_array = get_gumbel_softmax_sample(self.assigner_logit_array)
        abstract_agent_assignments = torch.argmax(one_hot_assignment_array, dim=-1)
        return abstract_agent_assignments