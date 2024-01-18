import torch
from torch import nn
from torch.nn import functional as F
from utils import get_gumbel_softmax_sample, get_linear_nonlinear_function, create_policy_network


class AbstractionModelJointPolicy(nn.Module):
    def __init__(self, state_space_dim, abs_action_space_dim, enc_hidden_dim, num_agents, num_abs_agents, action_space_dim=2):
        super(AbstractionModelJointPolicy, self).__init__()

        # Define the encoder, decoder, and assigner
        self.abstract_jointpolicy = Encoder(
            state_space_dim, abs_action_space_dim, enc_hidden_dim, num_abs_agents)
        self.ground_jointpolicy = Decoder(
            num_abs_agents, num_agents, abs_action_space_dim, action_space_dim)
        self.assigner = Assigner(num_abs_agents, num_agents)

    def forward(self, state):
        # Pass the state through the encoder to get abstract actions
        abs_actions = self.abstract_jointpolicy(state)

        # sample assigner to get assignments
        abstract_agent_assignments = self.assigner(state)

        # Pass the abstract actions ans assignments through the decoder to get action probabilities
        ground_action_probability_vectors = self.ground_jointpolicy(
            state, abs_actions, abstract_agent_assignments)

        return ground_action_probability_vectors


class Encoder(nn.Module):
    def __init__(self, state_space_dim, abs_action_space_dim, enc_hidden_dim, num_abs_agents):
        super(Encoder, self).__init__()
        self.state_space_dim = state_space_dim
        self.abs_action_space_dim = abs_action_space_dim
        self.hidden_dim = enc_hidden_dim
        self.num_abs_agents = num_abs_agents
        self.joint_abs_action_dim = abs_action_space_dim*num_abs_agents

        # realize the architecture
        independent_mode = True
        if independent_mode:
            enc_hidden_dim_per_policy = int(enc_hidden_dim/num_abs_agents)
            output_dimension = abs_action_space_dim
            num_of_policy_realizations = num_abs_agents
        else:
            enc_hidden_dim_per_policy = enc_hidden_dim
            output_dimension = abs_action_space_dim*num_abs_agents
            num_of_policy_realizations = 1
        self.abstract_agent_policy_networks = [create_policy_network(
            state_space_dim, enc_hidden_dim_per_policy, output_dimension) for idx in range(num_of_policy_realizations)]

    def abstract_agent_joint_policy(self, state, independent_mode=True):
        # print(state.shape)
        if independent_mode:
            list_of_abs_action_logit_vectors = [
                policy_network(state) for policy_network in self.abstract_agent_policy_networks]
            logit_array = torch.stack(list_of_abs_action_logit_vectors, dim=len(state.shape)-1)
        else:
            logit_array = self.abstract_agent_policy_networks[0](state)
            if len(state.shape)==2:
                logit_array = torch.reshape(logit_array, shape=(len(state), self.num_abs_agents, self.abs_action_space_dim))
            else:
                logit_array = torch.reshape(logit_array, shape=(self.num_abs_agents, self.abs_action_space_dim))
        return logit_array

    def forward(self, state, independent_mode=True):
        logit_array = self.abstract_agent_joint_policy(state, independent_mode=independent_mode)
        one_hot_array = get_gumbel_softmax_sample(logit_array)
        abstract_actions = torch.argmax(one_hot_array, dim=-1)
        return abstract_actions


class Decoder(nn.Module):
    def __init__(self, num_abs_agents, num_agents, abs_action_space_dim, action_space_dim=2):
        super(Decoder, self).__init__()
        self.num_agents = int(num_agents)

        # initialize ground agent embedding
        embedding_dim = 256
        self.groundagent_embedding = nn.Embedding(
            num_embeddings=num_agents, embedding_dim=embedding_dim)

        # initialize policies (input dimension is 1 (abstract action index) + embed_dim)
        state_space_dim = 1 + embedding_dim
        enc_hidden_dim = 256
        output_dimension = action_space_dim
        self.shared_action_policy_network = create_policy_network(
            state_space_dim, enc_hidden_dim, output_dimension)

    def forward(self, state, abs_actions, abstract_agent_assignments):
        assigned_abstract_actions = torch.stack([abs_actions[idx][abstract_agent_assignments[idx]] for idx in range(state.shape[0])],dim=0)#[:,:,0]
        repeat_dims = (state.shape[0],1,1) if len(state.shape)==2 else (1,1)
        # print(torch.unsqueeze(assigned_abstract_actions,-1).shape)
        # print(torch.unsqueeze(self.groundagent_embedding(torch.LongTensor(range(self.num_agents))),dim=0).repeat(repeat_dims).shape)
        input_tensor = torch.cat([
                                torch.unsqueeze(assigned_abstract_actions,dim=-1), 
                                torch.unsqueeze(self.groundagent_embedding(torch.LongTensor(range(self.num_agents))),dim=0).repeat(repeat_dims)
                                ], dim=-1)
        action_probability_vectors = F.softmax(
            self.shared_action_policy_network(input_tensor), dim=-1)
        return action_probability_vectors


class Assigner(nn.Module):
    def __init__(self, num_abs_agents, num_agents):
        super(Assigner, self).__init__()
        # components are abstract agent weights
        self.assigner_embedding_dim = num_abs_agents

        # initialize ground agent embedding
        self.abs_agent_assignment_embedding = nn.Embedding(
            num_embeddings=num_agents, embedding_dim=self.assigner_embedding_dim)
        self.assigner_logit_array = self.abs_agent_assignment_embedding(
            torch.LongTensor(range(num_agents)))

    def forward(self, state):
        repeat_dims = (state.shape[0],1,1) if len(state.shape)==2 else (1,1)
        one_hot_assignment_array = get_gumbel_softmax_sample(
            self.assigner_logit_array.repeat(repeat_dims)) #samples for each member of batch

        abstract_agent_assignments = torch.argmax(
            one_hot_assignment_array, dim=-1)
        return abstract_agent_assignments
