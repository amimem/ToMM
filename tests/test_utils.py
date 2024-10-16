import torch
import torch.nn as nn
from utils import MultiChannelNet, get_gumbel_softmax_sample

# Create an instance of MultiChannelNet
net = MultiChannelNet(n_channels=2, input_size=5, hidden_layer_width=10, n_hidden_layers=3, output_size=3)

# Create a random input tensor
input_tensor = torch.randn(2, 5)

# Test the forward pass
output_tensor = net.forward(input_tensor)
print(output_tensor.shape)  # Expected output: torch.Size([2, 2, 3])

# Test the output dimension when output_dim is specified
net_with_output_dim = MultiChannelNet(n_channels=2, input_size=5, hidden_layer_width=10, n_hidden_layers=3, output_size=3, output_dim=(2, 2, 3))
output_tensor_with_output_dim = net_with_output_dim.forward(input_tensor)
print(output_tensor_with_output_dim.shape)  # Expected output: torch.Size([2, 2, 2, 3])

# Test the output dimension when output_dim is not specified
net_without_output_dim = MultiChannelNet(n_channels=2, input_size=5, hidden_layer_width=10, n_hidden_layers=3, output_size=3)
output_tensor_without_output_dim = net_without_output_dim.forward(input_tensor)
print(output_tensor_without_output_dim.shape)  # Expected output: torch.Size([2, 2, 3])

# Test the get_gumbel_softmax_sample function
logit_vector = torch.randn(2, 3)
tau = 0.5
gumbel_softmax_sample = get_gumbel_softmax_sample(logit_vector, tau=tau)
print(gumbel_softmax_sample.shape)  # Expected output: torch.Size([2, 3])

# Test the get_gumbel_softmax_sample function for output differentiablity
logit_vector = torch.randn(2, 3)
tau = 0.5
gumbel_softmax_sample = get_gumbel_softmax_sample(logit_vector, tau=tau, hard=True)
print(gumbel_softmax_sample.shape)  # Expected output: torch.Size([2, 3])
# The output tensor is differentiable
gumbel_softmax_sample.sum().backward()
print(logit_vector.grad)
