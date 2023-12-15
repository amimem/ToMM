from torch import nn
from gumbel_partition.AutoEncoder import Encoder, Decoder

class GumbelPartitionModel(nn.Module):
    def __init__(self, state_space_dim, abs_action_space_dim, enc_hidden_dim, num_agents, num_abs_agents, action_space_dim=2):
        super(GumbelPartitionModel, self).__init__()
        
        # Define the encoder and decoder
        self.encoder = Encoder(state_space_dim, abs_action_space_dim, enc_hidden_dim, num_abs_agents)
        self.decoder = Decoder(num_abs_agents, num_agents, abs_action_space_dim, action_space_dim)

    def forward(self, state):
        # Pass the state through the encoder to get abstract actions
        abs_actions = self.encoder(state)
        
        # Pass the abstract actions through the decoder to get actions
        actions = self.decoder(abs_actions)
        
        return actions