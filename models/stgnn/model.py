import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import math
import modules
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

class STGNN(nn.Module):
    def __init__(self, num_layers, num_nodes, in_features, conv_channels, residual_channels, out_features, num_hops, dilation):
        super(STGNN, self).__init__()
        self.graph_conv = modules.GraphConvolution(in_features, conv_channels, num_hops)
        self.layer_norm1 = modules.LayerNorm([num_nodes, conv_channels])
        self.output_layer = modules.Output(num_nodes, conv_channels, out_features)
        self.adj = torch.randn(num_nodes, num_nodes)  # Adjacency matrix

        # Initialization of the two dilated inception layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.filter_convs.append(modules.TemporalConvolution(residual_channels, conv_channels, dilation_factor=dilation))
        self.gate_convs.append(modules.TemporalConvolution(residual_channels, conv_channels, dilation_factor=dilation))
        self.num_layers = num_layers
    def forward(self, x):
        for i in range(self.num_layers):
            filter = self.filter_convs[i](x) # dilated inception layer followed by a tanh
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x) # dilated inception layer followed by a sigmoid
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training) # to prevent overfitting

        #x = self.graph_conv(x, self.adj) # Graph convolution
        #x = self.layer_norm1(x) # Normalize after graph convolution
        #x = self.output_layer(x) # Output layer
        return x
