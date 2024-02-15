import torch
from torch import nn
from torch.nn import functional as F
from utils import get_gumbel_softmax_sample, MultiChannelNet


class STOMPnet(nn.Module):
    """
    A scalable theory of mind policy network

    Args:
        state_space_dim (int): Dimension of the state space
        abs_action_space_dim (int): Dimension of the abstract action space
        enc_hidden_dim (int): Dimension of the encoder's hidden layer
        num_agents (int): Number of agents
        num_abs_agents (int): Number of abstract agents
        action_space_dim (int, optional): Dimension of the action space. Defaults to 2.
    """
    def __init__(self, state_space_dim, abs_action_space_dim, enc_hidden_dim, num_agents, num_abs_agents, action_space_dim=2):
        super(STOMPnet, self).__init__()

        # Define the encoder, decoder, and assigner
        self.sample_from_abstractjointpolicy = Encoder(
            state_space_dim, abs_action_space_dim, enc_hidden_dim, num_abs_agents)
        self.groundjointpolicy = Decoder(
            num_abs_agents, num_agents, abs_action_space_dim, action_space_dim)
        self.assigner = Assigner(num_abs_agents, num_agents)

    def forward(self, state):
        """
        Forward pass of the STOMPnet

        Args:
            state (torch.Tensor): Input state tensor

        Returns:
            torch.Tensor: Output tensor representing the action probabilities
        """
        # Pass the state through the encoder to get abstract actions
        abs_actions = self.sample_from_abstractjointpolicy(state)

        # call assigner to sample assignments
        abstract_agent_assignments = self.assigner(state)

        # Pass the abstract actions ans assignments through the decoder to get action probabilities
        ground_action_logit_vectors = self.groundjointpolicy(
            state, abs_actions, abstract_agent_assignments)

        return ground_action_logit_vectors


class Encoder(nn.Module):
    """
    A multiagent network for abstract agents. Each abstract agent has its own policy network (as a channel in a MultiChannelNet).
    The encoder is responsible for sampling abstract actions.

    Args:
        state_space_dim (int): Dimension of the state space.
        abs_action_space_dim (int): Dimension of the abstract action space.
        enc_hidden_dim (int): Dimension of the hidden layer in the encoder.
        num_abs_agents (int): Number of abstract agents.

    Attributes:
        state_space_dim (int): Dimension of the state space.
        abs_action_space_dim (int): Dimension of the abstract action space.
        hidden_dim (int): Dimension of the hidden layer in the encoder.
        num_abs_agents (int): Number of abstract agents.
        abstract_agent_policy_networks (MultiChannelNet): Multi-channel neural network for abstract agents.

    """

    def __init__(self, state_space_dim, abs_action_space_dim, enc_hidden_dim, num_abs_agents):
        super(Encoder, self).__init__()
        self.state_space_dim = state_space_dim
        self.abs_action_space_dim = abs_action_space_dim
        self.hidden_dim = enc_hidden_dim
        self.num_abs_agents = num_abs_agents

        # realize the architecture
        self.abstract_agent_policy_networks = MultiChannelNet(
            n_channels=num_abs_agents,
            input_size=state_space_dim,
            hidden_layer_width=enc_hidden_dim,
            output_size=abs_action_space_dim
        )

    def forward(self, state):  # , independent_mode=True):
        """
        Forward pass of the encoder.

        Args:
            state (torch.Tensor): Input state.

        Returns:
            torch.Tensor: Abstract actions.

        """
        logit_vectors = self.abstract_agent_policy_networks(state)
        one_hot_vectors = get_gumbel_softmax_sample(logit_vectors)
        # convert from sparse (onehot) to dense (index) representation
        abstract_actions = torch.argmax(one_hot_vectors, dim=-1)
        return abstract_actions


class Decoder(nn.Module):
    """
    Decoder module for the STOMPnet model.

    Args:
        num_abs_agents (int): Number of abstract agents.
        num_agents (int): Number of ground agents.
        abs_action_space_dim (int): Dimension of the abstract action space.
        action_space_dim (int, optional): Dimension of the action space. Defaults to 2.
        hidden_layer_width (int, optional): Width of the hidden layer. Defaults to 256.
        agent_embedding_dim (int, optional): Dimension of the agent embedding. Defaults to 256.

    Attributes:
        num_agents (int): Number of ground agents.
        groundagent_embedding (nn.Embedding): Embedding for ground agents.
        shared_groundagent_policy_network (MultiChannelNet): Multi-channel neural network for ground agents.
    """

    def __init__(self, num_abs_agents, num_agents, abs_action_space_dim, action_space_dim=2, hidden_layer_width=256, agent_embedding_dim=256):
        super(Decoder, self).__init__()
        self.num_agents = int(num_agents)

        # initialize ground agent embedding, dims=(num_agents,agent_embedding_dim)
        self.groundagent_embedding = nn.Embedding(
            num_embeddings=num_agents, embedding_dim=agent_embedding_dim)

        # initialize policies (input dimension is 1 (abstract action index) + embed_dim)
        state_space_dim = 1 + agent_embedding_dim
        self.shared_groundagent_policy_network = MultiChannelNet(
            n_channels=1,
            input_size=state_space_dim,
            hidden_layer_width=hidden_layer_width,
            output_size=action_space_dim,
            output_dim=[action_space_dim]
        )

    def forward(self, state, abs_actions, abstract_agent_assignments):
        """
        Forward pass of the Decoder module.

        Args:
            state (torch.Tensor): Input state.
            abs_actions (torch.Tensor): Abstract actions.
            abstract_agent_assignments (torch.Tensor): Assignments of abstract agents to ground agents.

        Returns:
            torch.Tensor: Action logit vectors.
        """
        batch_flag = len(state.shape) >= 2
        assigned_abstract_actions = torch.stack(
            [abs_actions[idx][abstract_agent_assignments[idx]] for idx in range(state.shape[0])], dim=0) if batch_flag else abs_actions[abstract_agent_assignments]
        if batch_flag:
            parallel_input = torch.cat([
                torch.unsqueeze(assigned_abstract_actions, dim=-1),
                torch.unsqueeze(self.groundagent_embedding(torch.LongTensor(
                    range(self.num_agents))), dim=0).repeat((state.shape[0], 1, 1))
            ], dim=-1)
        else:
            parallel_input = torch.cat([
                torch.unsqueeze(assigned_abstract_actions, dim=-1),
                self.groundagent_embedding(torch.LongTensor(
                    range(self.num_agents)))
            ], dim=-1)
        action_logit_vectors = self.shared_groundagent_policy_network(
            parallel_input)
        return action_logit_vectors


class Assigner(nn.Module):
    """
    Assigns ground agents to abstract agents.

    Args:
        num_abs_agents (int): Number of abstract agents.
        num_agents (int): Number of ground agents.

    Attributes:
        assigner_embedding_dim (int): Dimension of the abstract agent weights.
        assigner_logit_array (nn.Embedding): Embedding for ground agent assignments.

    """

    def __init__(self, num_abs_agents, num_agents):
        super(Assigner, self).__init__()
        # components are abstract agent weights
        self.assigner_embedding_dim = num_abs_agents

        abs_agent_assignment_embedding = nn.Embedding(
            num_embeddings=num_agents, embedding_dim=self.assigner_embedding_dim)
        
        self.assigner_logit_array = abs_agent_assignment_embedding(
            torch.LongTensor(range(num_agents)))

    def forward(self, state):
        repeat_dims = (state.shape[0], 1, 1) if len(
            state.shape) == 2 else (1, 1)
        one_hot_assignment_array = get_gumbel_softmax_sample(
            self.assigner_logit_array.repeat(repeat_dims))  # samples for each member of batch

        abstract_agent_assignments = torch.argmax(
            one_hot_assignment_array, dim=-1)
        return abstract_agent_assignments
