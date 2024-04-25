import torch
from torch import nn
from torch.nn import functional as F
from utils import get_gumbel_softmax_sample, MultiChannelNet
import numpy as np


class STOMPnet2(nn.Module):
    """
    A scalable theory of mind policy network

    Args:
        state_space_dim (int): Dimension of the state space. N.B. this class assumes batched state input, i.e. dimension (batch_size,state_space_dim)
        abs_action_space_dim (int): Dimension of the abstract action space
        enc_hidden_dim (int): Dimension of the encoder's hidden layer
        num_agents (int): Number of agents
        num_abs_agents (int): Number of abstract agents
        action_space_dim (int, optional): Dimension of the action space. Defaults to 2.
        agent_embedding_dim (int, optional): Dimension of the agent embedding. Defaults to 2.
        num_codebooks (int, optional): Number of codebooks. Defaults to 10.

    Attributes:
        sample_from_abstract_joint_policy (Encoder): Encoder module
        ground_joint_policy (Decoder): Decoder module
        assigner (Assigner): Assigner module

    """

    def __init__(self, state_space_dim, dec_hidden_dim, num_agents, num_abs_agents, action_space_dim=2, agent_embedding_dim=2, n_hidden_layers=2):
        super(STOMPnet2, self).__init__()
        self.num_agents=num_agents
        self.action_space_dim=action_space_dim
        # Define the encoder, decoder, and assigner

        self.abstract_agent_policies = [Decoder(
            input_dim=state_space_dim+agent_embedding_dim,
            action_space_dim=action_space_dim, 
            n_hidden_layers=n_hidden_layers, 
            hidden_layer_width=dec_hidden_dim
        ) for n in range(num_abs_agents)]

        self.assigner = Assigner(num_abs_agents, num_agents)

        self.ground_agent_embedding = nn.Embedding(
            num_embeddings=num_agents, embedding_dim=agent_embedding_dim)


    def forward(self, state):
        """
        Forward pass of the STOMPnet

        Args:
            state (torch.Tensor): Input state tensor. dimensions (batch_size, state_pace_dim)

        Returns:
            torch.Tensor: Output tensor representing the action probabilities (batch_size, num_agents, action_space_dim)
        """

        # call assigner to sample assignments
        batch_size = state.shape[0]
        abstract_agent_assignments = self.assigner(batch_size)

        # abstract_agent_assignments: (batch_size, num_agents, num_abs_agents)
        # assigned_abstract_actions: (batch_size, num_agents, agent_embedding_dim)

        # Pass the abstract actions ans assignments through the decoder to get action probabilities
        ground_action_logit_vectors = torch.empty(batch_size,self.num_agents,self.action_space_dim)
        for bit in range(batch_size):
            for git in range(self.ground_agent_embedding.weight.shape[0]):
                embedding_vector = self.ground_agent_embedding.weight[git]
                assigned_policy = self.abstract_agent_policies[torch.where(abstract_agent_assignments[bit,git,:])[0].item()]
                # state=torch.unsqueeze(state,
                #                     dim=1).repeat((1,self.num_abs_gents,1))        
                concated_input = torch.cat([
                    embedding_vector,
                    state[bit],
                    ], dim=-1)
                ground_action_logit_vectors[bit,git,:]=assigned_policy(concated_input)
        return ground_action_logit_vectors


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

    def __init__(self, input_dim, action_space_dim, n_hidden_layers, hidden_layer_width):
        super(Decoder, self).__init__()

        # initialize policies. Input dimension is (batch_size, 1 (abstract action index) + embed_dim).
        self.shared_ground_agent_policy_network = MultiChannelNet(
            n_channels=1,
            input_size= input_dim,
            hidden_layer_width=hidden_layer_width,
            output_size=action_space_dim,
            n_hidden_layers=n_hidden_layers,
            # squeezes out the singleton channel dimension
            output_dim=[action_space_dim]
        )

    def forward(self, contextualized_agentvector):
        """
        Forward pass of the Decoder module.

        Args:
            assigned_abstract_actions (torch.Tensor): Abstract actions. dimensions (batch_size, num_agents)

        Returns:
            torch.Tensor: Action logit vectors.
        """
        action_logit_vectors = self.shared_ground_agent_policy_network(
            contextualized_agentvector)
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
        return one_hot_assignment_array
