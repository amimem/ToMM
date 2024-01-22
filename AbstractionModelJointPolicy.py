import torch
from torch import nn
from torch.nn import functional as F
from utils import get_gumbel_softmax_sample, MultiChannelNet


class AbstractionModelJointPolicy(nn.Module):
    def __init__(self, state_space_dim, abs_action_space_dim, enc_hidden_dim, num_agents, num_abs_agents, action_space_dim=2):
        super(AbstractionModelJointPolicy, self).__init__()

        # Define the encoder, decoder, and assigner
        self.sample_from_abstractjointpolicy = Encoder(
            state_space_dim, abs_action_space_dim, enc_hidden_dim, num_abs_agents)
        self.groundjointpolicy = Decoder(
            num_abs_agents, num_agents, abs_action_space_dim, action_space_dim)
        self.assigner = Assigner(num_abs_agents, num_agents)

    def forward(self, state):
        # Pass the state through the encoder to get abstract actions
        abs_actions = self.sample_from_abstractjointpolicy(state)

        # call assigner to sample assignments
        abstract_agent_assignments = self.assigner(state)

        # Pass the abstract actions ans assignments through the decoder to get action probabilities
        ground_action_logit_vectors = self.groundjointpolicy(
            state, abs_actions, abstract_agent_assignments)

        return ground_action_logit_vectors


class Encoder(nn.Module):
    # a multiagent network for abstract agents
    def __init__(self, state_space_dim, abs_action_space_dim, enc_hidden_dim, num_abs_agents):
        super(Encoder, self).__init__()
        self.state_space_dim = state_space_dim
        self.abs_action_space_dim = abs_action_space_dim
        self.hidden_dim = enc_hidden_dim
        self.num_abs_agents = num_abs_agents

        # realize the architecture
        self.abstractagent_policy_networks = MultiChannelNet(
            n_channels=num_abs_agents,
            input_size=state_space_dim,
            hidden_layer_width=enc_hidden_dim,
            output_size=abs_action_space_dim
        )

    def forward(self, state):  # , independent_mode=True):
        logit_vectors = self.abstractagent_policy_networks(state)
        one_hot_vectors = get_gumbel_softmax_sample(logit_vectors)
        # convert from sparse (onehot) to dense (index) representation
        abstract_actions = torch.argmax(one_hot_vectors, dim=-1)
        return abstract_actions


class Decoder(nn.Module):
    # a single agent network that conditions on the ground agent index
    def __init__(self, num_abs_agents, num_agents, abs_action_space_dim, action_space_dim=2, hidden_layer_width=256):
        super(Decoder, self).__init__()
        self.num_agents = int(num_agents)

        # initialize ground agent embedding
        embedding_dim = 256
        self.groundagent_embedding = nn.Embedding(
            num_embeddings=num_agents, embedding_dim=embedding_dim)

        # initialize policies (input dimension is 1 (abstract action index) + embed_dim)
        state_space_dim = 1 + embedding_dim
        n_hidden_layers = 2
        self.shared_groundagent_policy_network = MultiChannelNet(
            n_channels=1,
            input_size=state_space_dim,
            hidden_layer_width=hidden_layer_width,
            output_size=action_space_dim,
            output_dim = [action_space_dim]
        )

    def forward(self, state, abs_actions, abstract_agent_assignments):
        # map abstract actions to respective agents based on assignments and then concatenate with respective ground agent embedding vector
        assigned_abstract_actions = torch.stack(
            [abs_actions[idx][abstract_agent_assignments[idx]] for idx in range(state.shape[0])], dim=0)
        repeat_dims = (state.shape[0], 1, 1) if len(
            state.shape) == 2 else (1, 1)
        parallel_input = torch.cat([
            torch.unsqueeze(assigned_abstract_actions, dim=-1),
            torch.unsqueeze(self.groundagent_embedding(torch.LongTensor(
                range(self.num_agents))), dim=0).repeat(repeat_dims)
        ], dim=-1)
        action_logit_vectors = self.shared_groundagent_policy_network(
            parallel_input)
        return action_logit_vectors


class Assigner(nn.Module):
    # assigns ground agents to abstract agents
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
        repeat_dims = (state.shape[0], 1, 1) if len(
            state.shape) == 2 else (1, 1)
        one_hot_assignment_array = get_gumbel_softmax_sample(
            self.assigner_logit_array.repeat(repeat_dims))  # samples for each member of batch

        abstract_agent_assignments = torch.argmax(
            one_hot_assignment_array, dim=-1)
        return abstract_agent_assignments
