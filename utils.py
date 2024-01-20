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


class MultiChannelNet(nn.Module):
    # implements a 2D module array architecture. Parameters refer to architecture of individual channels
    def __init__(self, n_channels=1, input_size=10, hidden_layer_width=256, n_hidden_layers=2, output_size=10, output_dim=None):
        super(MultiChannelNet, self).__init__()
        self.module_array = nn.ModuleList(
            nn.ModuleList(
                [nn.Linear(input_size, hidden_layer_width)] +
                [nn.Linear(hidden_layer_width, hidden_layer_width) for h_layer_idx in range(n_hidden_layers)] +
                [nn.Linear(hidden_layer_width, output_size)]
            ) for channel_idx in range(n_channels))
        self.n_channels = n_channels
        self.n_hidden_layers = n_hidden_layers
        self.output_dim = output_dim

    def forward(self, state):
        logit_vectors = []
        for channel_idx in range(self.n_channels):
            x = torch.relu(self.module_array[channel_idx][0](state))
            for layer_idx in range(self.n_hidden_layers+1):
                x = torch.relu(self.module_array[channel_idx][layer_idx+1](x))
            logit_vectors.append(x)
        # assumes wants output_dim=(n_channels,output_size)
        if self.n_channels > 1:
            output = torch.stack(logit_vectors, dim=1) if len(
                state.shape) == 2 else torch.cat(logit_vectors)
        elif self.n_channels == 1:
            output = logit_vectors[0]
            if self.output_dim is not None:
                output = output.reshape(output.shape[:-1] + self.output_dim)
        # dims with batches: (batch_size,num_agents,num_actions); dims without batches: (num_agents,num_actions)
        return output


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
