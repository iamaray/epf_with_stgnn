import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import math
import modules
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


# class Transform(nn.Module):
#     def __init__(self):
#         super(Transform, self).__init__()
#         pass

#     def forward(self):
#         pass


# class PositionalEncoding(nn.Module):
#     def __init__(self):
#         super(PositionalEncoding, self).__init__()
#         pass

#     def forward(self):
#         pass


# class SGNN(nn.Module):
#     def __init__(self):
#         super(SGNN, self).__init__()
#         pass

#     def forward(self):
#         pass


# class GRU(nn.Module):
#     def __init__(self):
#         super(GRU, self).__init__()
#         pass


# class STGNNwithGRU(nn.Module):
#     def __init__(self, outfea):
#         super(STGNNwithGRU, self).__init__()
#         pass

#     def forward(self):
#         pass


class STGNN(nn.Module):
    def __init__(self, num_nodes, in_features, temporal_features, out_features, num_hops):
        super(STGNN, self).__init__()
        self.graph_conv = modules.GraphConvolution(in_features, temporal_features, num_hops)
        self.temp_conv = modules.TemporalConvolution(temporal_features)
        self.layer_norm1 = modules.LayerNorm([num_nodes, temporal_features])
        self.output_layer = modules.Output(num_nodes, temporal_features, out_features)
        self.layer_norm2 = modules.LayerNorm([num_nodes, out_features])
        self.adj = torch.randn(num_nodes, num_nodes)  # Adjacency matrix

    def forward(self, x):
        x = self.graph_conv(x, self.adj) # Graph convolution
        x = self.layer_norm1(x) # Normalize after graph convolution
        x = x.unsqueeze(1)  # Adjusting dimensions for 2D convolution if necessary
        x = self.temp_conv(x) # Temporal convolution
        x = self.layer_norm2(x)
        x = self.output_layer(x) # Output layer
        return x
