import torch
import numpy as np
from torch import nn

class GatingNetwork(nn.Module):
    """Class for the Gating Network included in the MANN architecture."""

    def __init__(self, input_size, output_size, hidden_size, dropout_probability):

        # Superclass constructor
        super(GatingNetwork, self).__init__()

        # Dimensions
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Parameters
        w0 = torch.zeros(self.hidden_size, self.input_size)
        w1 = torch.zeros(self.hidden_size, self.hidden_size)
        w2 = torch.zeros(self.output_size, self.hidden_size)
        b0 = torch.zeros(self.hidden_size, 1)
        b1 = torch.zeros(self.hidden_size, 1)
        b2 = torch.zeros(self.output_size, 1)

        # Intialization
        w0 = self.initialize_gn_weights(w0)
        w1 = self.initialize_gn_weights(w1)
        w2 = self.initialize_gn_weights(w2)

        # Make explicit that the parameters must be optimized in the learning process
        self.w0 = nn.Parameter(w0)
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.b0 = nn.Parameter(b0)
        self.b1 = nn.Parameter(b1)
        self.b2 = nn.Parameter(b2)

        # Activation funcitons and layers to be exploited in the forward call
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_probability)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):

        # Input processing
        x = self.dropout(x)

        # Layer 1
        H0 = torch.matmul(self.w0, x) + self.b0
        H0 = self.elu(H0)
        H0 = self.dropout(H0)

        # Layer 2
        H1 = torch.matmul(self.w1, H0) + self.b1
        H1 = self.elu(H1)
        H1 = self.dropout(H1)

        # Layer 3
        H2 = torch.matmul(self.w2, H1) + self.b2

        # Softmax to output normalized blending coefficients
        H2 = self.softmax(H2)

        return H2

    def initialize_gn_weights(self, w):
        """Initialize the Gating Network weights using uniform distribution."""

        bound = np.sqrt(6. / np.sum(w.shape[-2:]))
        return w.uniform_(-bound, bound)






