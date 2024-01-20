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


def compare_plot(output_filenames):
    num_datasets = len(output_filenames)
    data_list = [np.load(filename, allow_pickle=True).item()
                 for filename in output_filenames]
    fig, ax = pl.subplots(num_datasets, 3,figsize=(9,num_datasets*3))
    seed_idx = 0
    for dit, dataset in enumerate(data_list):
        actions = dataset["actions"][seed_idx]
        states = dataset["states"][seed_idx]
        first = (dit,0) if num_datasets>1 else 0
        second = (dit,1) if num_datasets>1 else 1
        third = (dit,2) if num_datasets>1 else 2
        num_agents = len(actions[0])
        ax[first].imshow(actions.T, extent=[
                          0, actions.shape[0], 0, actions.shape[1]], aspect='auto')
        # for epsiode_change_time in np.where(np.array(dataset["times"])==dataset["T"])[0]:
        # ax[dit,0].axvline(epsiode_change_time)
        ax[first].set_ylabel(dataset['sys_parameters']['jointagent_groundmodel_paras']['groundmodel_name']+'\n\nagent index')
        if dit == len(data_list)-1:
            ax[first].set_xlabel('time index')
        else:
            ax[first].set_title('actions')

        ax[second].plot(np.linalg.norm(states, axis=1), '.')
        # for epsiode_change_time in np.where(np.array(dataset["times"])==dataset["T"])[0]:
        # ax[dit,1].axvline(epsiode_change_time)
        if dit == len(data_list)-1:
            ax[second].set_xlabel('time index')
        else:
            ax[second].set_title('state norm')

        corr_matrix = get_corr_matrix(actions)
        shift=-0.5
        p=ax[third].imshow(corr_matrix, extent=[
                          shift, num_agents+shift, shift, num_agents+shift])
        ax[third].set_xticks(range(num_agents))
        ax[third].set_yticks(range(num_agents))
        ax[third].set_xlim(shift, num_agents+shift)
        ax[third].set_ylim(shift, num_agents+shift)
        ax[third].set_ylabel('agent index')
        if dit == 1:
            ax[third].set_xlabel('agent index')
        if dit == 0:
            ax[third].set_title('action correlations')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax[third])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(p, cax=cax, orientation='vertical')
        # if dit == 0:
        #     fig.suptitle(r"ground model: $\rho="+str(
        #         dataset['baseline_paras']['corr'])+r"$ "+dataset['baseline_paras']['gen_type']+" ensemble")
    fig.tight_layout()
    return fig

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
