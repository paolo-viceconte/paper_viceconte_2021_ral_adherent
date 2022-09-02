import torch
import numpy as np
from torch import nn


class MotionPredictionNetwork(nn.Module):
    """Class for the Motion Prediction Network included in the MANN architecture."""

    def __init__(self, num_experts, input_size, output_size, hidden_size, dropout_probability):
        """Motion Prediction Network constructor."""

        # Superclass constructor
        super(MotionPredictionNetwork, self).__init__()

        # Dimensions
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        # Parameters
        w0 = torch.zeros(self.num_experts, self.hidden_size, self.input_size)
        w1 = torch.zeros(self.num_experts, self.hidden_size, self.hidden_size)
        w2 = torch.zeros(self.num_experts, self.output_size, self.hidden_size)
        b0 = torch.zeros(self.num_experts, self.hidden_size, 1)
        b1 = torch.zeros(self.num_experts, self.hidden_size, 1)
        b2 = torch.zeros(self.num_experts, self.output_size, 1)

        # Initialization
        w0 = self.initialize_mpn_weights(w0)
        w1 = self.initialize_mpn_weights(w1)
        w2 = self.initialize_mpn_weights(w2)

        # Make explicit that the parameters must be optimized in the learning process
        self.w0 = nn.Parameter(w0)
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.b0 = nn.Parameter(b0)
        self.b1 = nn.Parameter(b1)
        self.b2 = nn.Parameter(b2)

        # Activation functions and layers to be exploited in the forward call
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, x, blending_coefficients):
        """Motion Prediction Network 3-layers architecture."""

        # Input processing
        x = torch.unsqueeze(x, -1)
        x = self.dropout(x)

        # Layer 1
        blended_w0 = torch.einsum("ij,ikl->jkl", blending_coefficients, self.w0)
        blended_b0 = torch.einsum("ij,ikl->jkl", blending_coefficients, self.b0)
        H0 = torch.matmul(blended_w0, x) + blended_b0
        H0 = self.elu(H0)
        H0 = self.dropout(H0)

        # Layer 2
        blended_w1 = torch.einsum("ij,ikl->jkl", blending_coefficients, self.w1)
        blended_b1 = torch.einsum("ij,ikl->jkl", blending_coefficients, self.b1)
        H1 = torch.matmul(blended_w1, H0) + blended_b1
        H1 = self.elu(H1)
        H1 = self.dropout(H1)

        # Layer 3
        blended_w2 = torch.einsum("ij,ikl->jkl", blending_coefficients, self.w2)
        blended_b2 = torch.einsum("ij,ikl->jkl", blending_coefficients, self.b2)
        H2 = torch.matmul(blended_w2, H1) + blended_b2
        H2 = torch.squeeze(H2, -1)

        return H2

    @staticmethod
    def initialize_mpn_weights(w):
        """Initialize the Motion Prediction Network weights using uniform distribution
        with bounds defined on the basis of the dimensions of the network layers."""

        bound = np.sqrt(6. / np.prod(w.shape[-2:]))
        return w.uniform_(-bound, bound)

