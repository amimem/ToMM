import torch.nn.functional as F
import torch.nn as nn

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