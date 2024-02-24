import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as pl
import numpy as np
import seaborn as sb

def get_gumbel_softmax_sample(logit_vector, tau=1):
    """
    Compute Gumbel softmax sample for both hard and soft cases.

    Args:
        logit_vector (torch.Tensor): The input logit vector.
        tau (float, optional): The temperature parameter for Gumbel softmax. Defaults to 1.

    Returns:
        torch.Tensor: The differentiable version of the hard Gumbel softmax sample.
    """
    y_hard = F.gumbel_softmax(logit_vector, tau=tau, hard=True)
    y_soft = F.gumbel_softmax(logit_vector, tau=tau, hard=False)
    y_hard_diff = y_hard - y_soft.detach() + y_soft
    return y_hard_diff


class MultiChannelNet(nn.Module):
    """
    Implements a 2D module array architecture. Each row is a channel. Parameters refer to architecture of individual channels.
    Dimensions of output: (n_channels, output_size), unless specified using input argument output_dim.

    Args:
        n_channels (int, optional): Number of channels. Defaults to 1.
        input_size (int, optional): Size of input vector. Defaults to 10.
        hidden_layer_width (int, optional): Width of hidden layers. Defaults to 256.
        n_hidden_layers (int, optional): Number of hidden layers. Defaults to 2.
        output_size (int, optional): Size of output vector. Defaults to 10.
        output_dim (tuple, optional): Dimensions of output. Defaults to None.
    """

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
        self.n_all_layers = n_hidden_layers + 2 # including input and output layers
        self.default_output_dim = (n_channels, output_size)
        self.output_dim = self.default_output_dim if output_dim is None else output_dim        

                
    def forward(self, state):
            """
            Forward pass of the model.

            Args:
                state (torch.Tensor): Input state tensor.

            Returns:
                torch.Tensor: Output tensor after passing through the model.
            """
            logit_vectors = []
            for channel_idx in range(self.n_channels):
                x = state.clone() # clone to avoid in-place operations
                for layer_idx in range(0, self.n_all_layers):
                    x = torch.relu(self.module_array[channel_idx][layer_idx](x))
                logit_vectors.append(x) 
            if len(state.shape) >= 2:
                output = torch.stack(logit_vectors, dim=-2)
            else:
                output = torch.stack(logit_vectors)

            if self.output_dim != self.default_output_dim:
                output = output.reshape(tuple(output.shape[:-2]) + tuple(self.output_dim))
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
            
        # ax[second].plot(np.linalg.norm(states, axis=1), '.')
        datatmp = np.linalg.norm(np.diff(states,axis=0), axis=1)
        epsiode_length = 20
        for start_idx,col in enumerate(sb.color_palette('viridis',epsiode_length)):#episode length
            x=datatmp[1+start_idx::epsiode_length]
            y=datatmp[start_idx:-2:epsiode_length]
            ax[second].plot(x[:len(y)],y, '.',color=col)
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
