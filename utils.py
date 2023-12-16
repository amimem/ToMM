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

def compare_plot(data_pair):
    fig,ax = pl.subplots(2,3)
    for dit,dataset in enumerate(data_pair):
        ax[dit,0].imshow(np.array(dataset["actions"]))
        for epsiode_change_time in np.where(np.array(dataset["times"])==dataset["T"])[0]:
            ax[dit,0].axvline(epsiode_change_time)
        ax[dit,0].set_ylabel('agent index')
        ax[dit,0].set_xlabel('time index')

        ax[dit,1].plot(np.linalg.norm(np.array(dataset["states"]),axis=1))
        for epsiode_change_time in np.where(np.array(dataset["times"])==dataset["T"])[0]:
            ax[dit,1].axvline(epsiode_change_time)
        ax[dit,1].set_ylabel('state index')
        ax[dit,1].set_xlabel('time index')

def get_corr_matrix(index_seq, n_agents):
    n_steps = len(index_seq)
    sims = index2binary(index_seq, n_agents)
    corr_matrix = np.zeros([n_agents] * 2)
    for i in range(n_agents):
        for j in range(n_agents):
            if i < j:
                corr_matrix[i, j] = 2 * \
                    np.sum(sims[i, :] == sims[j, :]) / n_steps - 1
    return corr_matrix + corr_matrix.T + np.identity(n_agents)