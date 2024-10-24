import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as pl
import numpy as np
import seaborn as sb
# from MatchNet import MatchNet

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

def numpy_scalar_to_python(value):
    if isinstance(value, np.generic):
        return value.item()
    return value

def count_parameters(model):
    assert isinstance(model, nn.Module), "model must be a torch.nn.Module"
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MultiChannelNet(nn.Module):
    """
    Implements a 2D module array architecture. Each row is a channel. Parameters refer to architecture of individual channels.
    Dimensions of output: (batch_size, n_channels, output_size), unless specified using input argument output_dim.

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
            state (torch.Tensor): Input state tensor of dimension (batch_size, statespace_dim).

        Returns:
            torch.Tensor: Output tensor after passing through the model. 
            dimension (batch_size, nchannels, output size) for nchannels>1
                      (batch_size, output size) if channels = 1
        """
        logit_vectors = []
        for channel_idx in range(self.n_channels):
            x = state.clone() # clone to avoid in-place operations
            for layer_idx in range(0, self.n_all_layers):
                x = torch.relu(self.module_array[channel_idx][layer_idx](x))
            logit_vectors.append(x)
        output = logit_vectors[0] if self.n_channels == 1 else torch.stack(logit_vectors, dim=1)
        if self.output_dim != self.default_output_dim:
            output = output.reshape(tuple(output.shape[:-1]) + tuple(self.output_dim))
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


def solve_quadratic(a,b,c):
    return (-b+np.sqrt(b**2-4*a*c))/(2*a) #assumes negative root always extraneous

def get_width(config):

    model_name = config['model_name']
    num_agents = config['num_agents']
    state_space_dim = config['state_space_dim']
    action_space_dim = config['action_space_dim']
    n_features = config['n_features']
    n_hidden_layers = config['n_hidden_layers']
    num_abs_agents = config['num_abs_agents']
    num_params = config['num_parameters']
    abs_action_space_dim = config['abs_action_space_dim']

    if model_name=='single':
        a=n_hidden_layers*num_agents
        b=num_agents*(action_space_dim+state_space_dim)
        c=-num_params
        W=solve_quadratic(a,b,c)
    elif model_name=='multi':
        a=n_hidden_layers
        b=num_agents*action_space_dim+state_space_dim
        c=-num_params
        W=solve_quadratic(a,b,c)
    elif model_name=='stomp':
        enc2dec_ratio = config['enc2dec_ratio']
        a=(num_abs_agents+enc2dec_ratio**2)*n_hidden_layers
        b=num_abs_agents*(abs_action_space_dim+state_space_dim) + (n_features+abs_action_space_dim+action_space_dim)*enc2dec_ratio
        c=num_agents*(num_abs_agents + n_features) -num_params
        W=solve_quadratic(a,b,c)
    elif model_name=='stomp2':
        a=num_abs_agents*n_hidden_layers
        b=num_abs_agents*(n_features+abs_action_space_dim+action_space_dim)
        c=num_agents*(num_abs_agents + n_features) -num_params
        W=solve_quadratic(a,b,c)
    elif model_name=='decoderonly':
        enc2dec_ratio = config['enc2dec_ratio']
        a=(enc2dec_ratio**2)*n_hidden_layers
        b= (n_features+state_space_dim+action_space_dim)*enc2dec_ratio 
        c=num_agents*n_features-num_params
        W=solve_quadratic(a,b,c)
    elif model_name=='match':
        # a=n_hidden_layers*num_abs_agents
        # b= num_abs_agents*(n_features+state_space_dim) 
        a=(n_hidden_layers+1)*num_abs_agents
        b= num_abs_agents*(state_space_dim) 
        c=num_agents*num_abs_agents-num_params
        W=solve_quadratic(a,b,c)
    else:
        print('choose valid model name')
    return W

# def get_untrained_net(training_dict,data_settings,hidden_dim,model_name):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     print(f"Using {device} device", flush=True)
#     args=training_dict.copy()
#     action_space_dim = 2
#     state_space_dim = data_settings['K']
#     num_agents = int(data_settings['N'])
#     # Initialize abstraction system model
#     n_hidden_layers = args["n_hidden_layers"]
#     if model_name == 'stomp':
#         # to match the M generated by joint policy of ground model set as: data_settings['sys_parameters']['jointagent_groundmodel_paras']['M']
#         net = STOMPnet(
#             state_space_dim = state_space_dim,
#             abs_action_space_dim = args["L"],
#             enc_hidden_dim = hidden_dim,
#             dec_hidden_dim = int(args["enc2dec_ratio"]*hidden_dim),
#             num_agents=num_agents,
#             num_abs_agents = args["M"],
#             action_space_dim=action_space_dim,
#             agent_embedding_dim= args["n_features"],
#             n_hidden_layers=n_hidden_layers,
#             num_codebooks=args["num_codebooks"]
#         )
#     elif model_name == 'stomp2':
#         # to match the M generated by joint policy of ground model set as: data_settings['sys_parameters']['jointagent_groundmodel_paras']['M']
#         net = STOMPnet2(
#             state_space_dim = state_space_dim, 
#             dec_hidden_dim = hidden_dim, 
#             num_agents=num_agents, 
#             num_abs_agents = args["M"], 
#             action_space_dim=action_space_dim, 
#             agent_embedding_dim=args["n_features"], 
#             n_hidden_layers=n_hidden_layers
#         )
#     elif model_name == 'single':
#         net = MultiChannelNet(
#             n_channels=num_agents,
#             input_size=state_space_dim,
#             hidden_layer_width= hidden_dim,
#             n_hidden_layers=n_hidden_layers,
#             output_size=action_space_dim
#         )
#     elif model_name == 'multi':
#         net = MultiChannelNet(
#             n_channels=1,
#             input_size=state_space_dim,
#             hidden_layer_width= hidden_dim,
#             n_hidden_layers=n_hidden_layers,
#             output_size=num_agents*action_space_dim,
#             output_dim=(num_agents, action_space_dim)
#         )
#     elif model_name == 'decoderonly':
#         #feeds state directly into decoder by overloading the assigned abstract action input
#         net = Decoder(
#             num_agents=num_agents, 
#             abs_action_space_dim=state_space_dim, 
#             action_space_dim=action_space_dim, 
#             agent_embedding_dim=args["n_features"], 
#             n_hidden_layers=n_hidden_layers, 
#             hidden_layer_width=int(args["enc2dec_ratio"]*hidden_dim)
#         )
#     elif model_name == 'match':
#         net = MatchNet(
#             state_dim=state_space_dim,
#             hidden_size=hidden_dim,
#             num_ground_agents=num_agents,
#             num_abstract_agents=args["M"],
#             num_layers=n_hidden_layers
#         )
#     else:
#         print('choose valid model', flush=True)
#     net.to(device)
#     return net
    