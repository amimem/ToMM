import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as pl
import numpy as np


def get_gumbel_softmax_sample(logit_vector, tau=1):
    # Compute Gumbel softmax sample for both hard and soft cases
    y_hard = F.gumbel_softmax(logit_vector, tau=tau, hard=True)
    y_soft = F.gumbel_softmax(logit_vector, tau=tau, hard=False)

    # Create a differentiable version of y_hard
    y_hard_diff = y_hard - y_soft.detach() + y_soft

    return y_hard_diff


def get_linear_nonlinear_function(input_dim, output_dim):
    # Create a linear layer
    linear_layer = nn.Linear(input_dim, output_dim)

    # Define the function to be returned
    def nonlinear_function(input_tensor):
        return F.sigmoid(linear_layer(input_tensor))

    return nonlinear_function

# Define the neural network architecture type, E.g.
def create_policy_network(state_space_dim, enc_hidden_dim, output_dim):
    fc1 = nn.Linear(state_space_dim, enc_hidden_dim)
    fc2 = nn.Linear(enc_hidden_dim, output_dim)

    def policy(state, fc1=fc1, fc2=fc2):
        x = F.relu(fc1(state))
        logits = fc2(x)
        return logits
    return policy

class JointPolicyNet(nn.Module):  
    def __init__(self, n_input, n_hidden, n_out, n_channels, n_hidden_layers):
        super(JointPolicyNet, self).__init__()
        assert (n_out/n_channels).is_integer(), "number of outputs/number of channels should be integer-valued"
        assert (n_hidden/n_channels).is_integer(), f"hidden layer width{n_hidden}/number of channels{n_channels} should be integer-valued"
        n_hidden = int(n_hidden/n_channels)

        self.module_array = nn.ModuleList(
            nn.ModuleList(
                [nn.Linear(n_input, n_hidden)] +
                [nn.Linear(n_hidden, n_hidden) for h_layer_idx in range(n_hidden_layers)] +
                [nn.Linear(n_hidden, n_out)]
            ) for channel_idx in range(n_channels))

        self.n_channels = n_channels
        self.n_hidden_layers = n_hidden_layers
       
    def forward(self, state):
        logit_vectors = []
        for channel_idx in range(self.n_channels):
            x = torch.relu(self.module_array[channel_idx][0](state))
            for layer_idx in range(self.n_hidden_layers+1):
                x = torch.relu(self.module_array[channel_idx][layer_idx+1](x))
            logit_vectors.append(x)
        if self.n_channels > 1:
            output = torch.stack(logit_vectors, dim=1 if len(state.shape) == 2 else 0)
        else:
            output = logit_vectors[0]
        return output #dims with batches: (batch_size,num_agents,num_actions); dims without batches: (num_agents,num_actions)

def compare_plot(pair_of_output_filenames, output_dir='output/'):
    data_pair = [np.load(filename, allow_pickle=True).item()
                 for filename in pair_of_output_filenames]
    fig, ax = pl.subplots(2, 3)

    for dit, dataset in enumerate(data_pair):
        actions = np.array(dataset["actions"])
        states = np.array(dataset["states"])
        num_agents = len(actions[0])
        ax[dit, 0].imshow(actions.T, extent=[
                          0, actions.shape[0], 0, actions.shape[1]], aspect='auto')
        # for epsiode_change_time in np.where(np.array(dataset["times"])==dataset["T"])[0]:
        # ax[dit,0].axvline(epsiode_change_time)
        ax[dit, 0].set_ylabel(dataset['model_name']+'\n\nagent index')
        if dit == 1:
            ax[dit, 0].set_xlabel('time index')
        else:
            ax[dit, 0].set_title('actions')

        ax[dit, 1].plot(np.linalg.norm(states, axis=1), '.')
        # for epsiode_change_time in np.where(np.array(dataset["times"])==dataset["T"])[0]:
        # ax[dit,1].axvline(epsiode_change_time)
        if dit == 1:
            ax[dit, 1].set_xlabel('time index')
        else:
            ax[dit, 1].set_title('state norm')

        corr_matrix = get_corr_matrix(dataset["actions"])
        ax[dit, 2].imshow(corr_matrix, extent=[
                          0.5, num_agents+0.5, 0.5, num_agents+0.5])
        ax[dit, 2].set_xticks([1]+list(ax[dit, 2].get_xticks()))
        ax[dit, 2].set_yticks([1]+list(ax[dit, 2].get_yticks()))
        ax[dit, 2].set_xlim(0.5, num_agents+0.5)
        ax[dit, 2].set_ylim(0.5, num_agents+0.5)
        ax[dit, 2].set_ylabel('agent index')
        if dit == 1:
            ax[dit, 2].set_xlabel('agent index')

        if dit == 0:
            ax[dit, 2].set_title('action correlations')
        if dit == 0:
            fig.suptitle(r"ground model: $\rho="+str(
                dataset['baseline_paras']['corr'])+r"$ "+dataset['baseline_paras']['gen_type']+" ensemble")
    fig.tight_layout()
    fig.savefig(f'{pair_of_output_filenames[0][:-4]}_summary_fig.pdf', transparent=True, bbox_inches="tight", dpi=300)

def get_corr_matrix(action_seq):
    num_agents = len(action_seq[0])
    num_steps = len(action_seq)
    sims = np.array(action_seq).T
    corr_matrix = np.zeros([num_agents] * 2)
    for i in range(num_agents):
        for j in range(num_agents):
            if i < j:
                corr_matrix[i, j] = 2 * \
                    np.sum(sims[i, :] == sims[j, :]) / num_steps - 1
    return corr_matrix + corr_matrix.T + np.identity(num_agents)
