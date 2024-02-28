import torch
from torch import nn
from torch.nn import functional as F
from utils import get_gumbel_softmax_sample, MultiChannelNet


class STOMPnet(nn.Module):
    """
    A scalable theory of mind policy network

    Args:
        state_space_dim (int): Dimension of the state space. N.B. this class assumes batched state input, i.e. dimension (batch_size,state_space_dim)
        abs_action_space_dim (int): Dimension of the abstract action space
        enc_hidden_dim (int): Dimension of the encoder's hidden layer
        num_agents (int): Number of agents
        num_abs_agents (int): Number of abstract agents
        action_space_dim (int, optional): Dimension of the action space. Defaults to 2.

    Attributes:
        sample_from_abstract_joint_policy (Encoder): Encoder module
        ground_joint_policy (Decoder): Decoder module
        assigner (Assigner): Assigner module

    """

    def __init__(self, state_space_dim, abs_action_space_dim, enc_hidden_dim, num_agents, num_abs_agents, action_space_dim=2):
        super(STOMPnet, self).__init__()

        # Define the encoder, decoder, and assigner
        self.sample_from_abstract_joint_policy = Encoder(
            state_space_dim, abs_action_space_dim, enc_hidden_dim, num_abs_agents)
        self.ground_joint_policy = Decoder(num_agents, action_space_dim)
        self.assigner = Assigner(num_abs_agents, num_agents)

    def forward(self, state):
        """
        Forward pass of the STOMPnet

        Args:
            state (torch.Tensor): Input state tensor. dimensions (batch_size, state_pace_dim)

        Returns:
            torch.Tensor: Output tensor representing the action probabilities (batch_size, num_agents, action_space_dim)
        """
        # Pass the state through the encoder to get abstract actions
        abs_actions = self.sample_from_abstract_joint_policy(state)

        # call assigner to sample assignments
        batch_size = state.shape[0]
        abstract_agent_assignments = self.assigner(batch_size)

        # Pass the abstract actions ans assignments through the decoder to get action probabilities
        ground_action_logit_vectors = self.ground_joint_policy(
            abs_actions, abstract_agent_assignments)

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
            state (torch.Tensor): Input state. dimensions (batch_size, state_space_dim)

        Returns:
            torch.Tensor: Abstract actions. dimensions (batch_size, num_abs_agents)

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
        num_agents (int): Number of ground agents.
        action_space_dim (int, optional): Dimension of the action space. Defaults to 2.
        hidden_layer_width (int, optional): Width of the hidden layer. Defaults to 256.
        agent_embedding_dim (int, optional): Dimension of the agent embedding. Defaults to 256.

    Attributes:
        num_agents (int): Number of ground agents.
        ground_agent_embedding (nn.Embedding): Embedding for ground agents.
        shared_ground_agent_policy_network (MultiChannelNet): Multi-channel neural network for ground agents.
    """

    def __init__(self, num_agents, action_space_dim=2, hidden_layer_width=256, agent_embedding_dim=256):
        super(Decoder, self).__init__()
        self.num_agents = int(num_agents)

        # initialize ground agent embedding, dims=(num_agents,agent_embedding_dim)
        self.ground_agent_embedding = nn.Embedding(
            num_embeddings=num_agents, embedding_dim=agent_embedding_dim)

        # initialize policies. Input dimension is (batch_size, 1 (abstract action index) + embed_dim).
        state_space_dim = 1 + agent_embedding_dim
        self.shared_ground_agent_policy_network = MultiChannelNet(
            n_channels=1,
            input_size=state_space_dim,
            hidden_layer_width=hidden_layer_width,
            output_size=action_space_dim,
            # squeezes out the singleton channel dimension
            output_dim=[action_space_dim]
        )

    def forward(self, abs_actions, abstract_agent_assignments):
        """
        Forward pass of the Decoder module.

        Args:
            abs_actions (torch.Tensor): Abstract actions. dimensions (batch_size, num_abs_agents)
            abstract_agent_assignments (torch.Tensor): Assignments of ground agents to abstract agents. ()

        Returns:
            torch.Tensor: Action logit vectors.
        """
        batch_size = abs_actions.shape[0]
        assigned_abstract_actions = torch.stack(
            [abs_actions[idx][abstract_agent_assignments[idx]] for idx in range(batch_size)], dim=0)

        # run decoder network in parallel over all ground agents
        repeat_dims = (batch_size, 1, 1)
        parallel_input = torch.cat([
            torch.unsqueeze(assigned_abstract_actions, dim=-1),
            torch.unsqueeze(self.ground_agent_embedding.weight,
                            dim=0).repeat(repeat_dims)
        ], dim=-1)
        action_logit_vectors = self.shared_ground_agent_policy_network(
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

        self.abs_agent_assignment_embedding = nn.Embedding(
            num_embeddings=num_agents, embedding_dim=self.assigner_embedding_dim)

    def forward(self, batch_size):
        # samples for each member of batch
        repeat_dims = (batch_size, 1, 1)
        one_hot_assignment_array = get_gumbel_softmax_sample(
            self.abs_agent_assignment_embedding.weight.repeat(repeat_dims))

        abstract_agent_assignments = torch.argmax(
            one_hot_assignment_array, dim=-1)
        return abstract_agent_assignments
