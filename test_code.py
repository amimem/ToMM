import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import logging
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)

class Encoder(nn.Module):
    def __init__(self, state_dim, abs_action_space_dim, hidden_dim, num_abstract_agents):
        super().__init__()
        self.state_dim = state_dim
        self.abs_action_space_dim = abs_action_space_dim
        self.hidden_dim = hidden_dim
        self.joint_abs_action_dim=abs_action_space_dim*num_abstract_agents

        # Define the neural network architecture, E.g.
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.joint_abs_action_dim)

    def forward(self, state, hard_flag=True):
        logging.info(f"Input state: {state}")
        x = F.relu(self.fc1(torch.Tensor([state])))
        logits = self.fc2(x)

        # do gumbel softmax for each abstract agent and collect results
        gumbel_softmax_index_samples = torch.empty(num_abstract_agents, abs_action_space_dim, dtype=torch.long)
        for abs_agent_id in range(num_abstract_agents):
            abs_agent_logits = logits[abs_agent_id*abs_action_space_dim:(abs_agent_id+1)*abs_action_space_dim]
            # Apply Gumbel-Softmax relaxation to the logits to obtain action probabilities
            gumbel_dist = distributions.Gumbel(0, 1)
            gumbel_noise = gumbel_dist.sample(abs_agent_logits.shape)
            one_hot_vector = F.gumbel_softmax(abs_agent_logits + gumbel_noise, tau=1, hard=hard_flag)
            gumbel_softmax_index_samples[abs_agent_id] = torch.where(one_hot_vector)[0]
        logging.info(f"Output gumbel_softmax_index_samples: {gumbel_softmax_index_samples}")
        return gumbel_softmax_index_samples

class Decoder(nn.Module):
    def __init__(self, abs_action_space_dimension, action_dim):
        super(Decoder, self).__init__()
        self.decode_map=torch.randint(low=0,high=action_dim, size=(num_abstract_agents, abs_action_space_dim, agents_per_abs_agent))
        logging.info(f"self.decode_map: {self.decode_map}")
    def decode(self, abs_actions):
        logging.info(f"Input abs_actions: {abs_actions}")
        output_actions=[]
        for abs_agent_id, abs_action in enumerate(abs_actions):
            output_actions+=list(self.decode_map[abs_agent_id,abs_action,:])
        logging.info(f"Output output_actions: {output_actions}")
        return output_actions

class GumbelPartitionModel(nn.Module):
    def __init__(self, state_dim, abs_action_space_dim, hidden_dim, num_abstract_agents, action_dim=2):
        super().__init__()
        self.encoder = Encoder(
            state_dim,
            abs_action_space_dim,
            hidden_dim,
            num_abstract_agents
            )
        self.decoder = Decoder(
            abs_action_space_dim,
            action_dim)

    def forward(self, state, hard_flag=True):
        abs_actions = self.encoder.forward(state, hard_flag)
        actions = self.decoder.decode(abs_actions)
        return actions

writer = SummaryWriter()

num_abstract_agents = 2
agents_per_abs_agent = 2

#combine in partition
partition = []
for abs_agent_id in range(num_abstract_agents):
    partition+=[abs_agent_id]*agents_per_abs_agent

# run model given partition
state_dim = 1
state_space_dim = 10
abs_action_space_dim = 10
hidden_dim = 10
num_abstract_agents = partition[-1]+1
action_dim = 2

model = GumbelPartitionModel(
    state_dim,
    abs_action_space_dim,
    hidden_dim,
    num_abstract_agents,
    action_dim=2
    )

if __name__ == '__main__':
    num_steps = 5
    for step in range(num_steps):
        logging.info(f"Step: {step}")
        state = np.random.randint(state_space_dim)
        writer.add_graph(model, torch.Tensor([state]))
        writer.close()
        logging.info(f"Input state: {state}")
        actions = model.forward(state, hard_flag=False)
