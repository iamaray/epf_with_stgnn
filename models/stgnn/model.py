from .layer_constr_helpers import *
from .modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import math
from torch_geometric.data import Data

device = "cuda" if torch.cuda.is_available() else "cpu"


class STGNN(nn.Module):
    def __init__(
        self,
        K: int = 0,
        beta: float = 0.05,
        c_in: int = 14,
        num_nodes: int = 3,
        batch_size: int = 32,
        sequence_length: int = 168,
        residual_chans: int = 32,
        conv_chans: int = 32,
        gc_support_len: int = 3,
        gc_order: int = 2,
        gc_dropout: float = 0.2,
        skip_chans: int = 64,
        end_chans: int = 128,
        num_layers: int = 10,
        dilation_multiplier: int = 1,
        pred_length: int = 24,
        dropout_factor: float = 0.2,
        use_graph_conv: bool = True,
        use_temp_conv: bool = True,
        use_diffusion: bool = True,
        adj_type: str = 'learned'
    ):
        """
        Initializes the STGNN model.

        Parameters:
        Dependent on input data:
        - c_in: Number of input channels (x.shape[1]).
        - num_nodes: Number of nodes in the graph (x.shape[2], y.shape[2]).
        - sequence_length: Length of the input sequence (x.shape[3]).

        - residual_chans: Number of residual channels.
        - conv_chans: Number of convolution channels.
        - skip_chans: Number of skip channels.
        - end_chans: Number of end channels.
        - num_layers: Number of layers in the model.
        - K, beta: Parameters for Graph Convolution layers.
        - dilation_multiplier: Dilation multiplier for Temporal Convolution layers.
        - pred_length: Length of the prediction (y.shape[1]).
        - dropout_factor: Dropout factor for dropout function.
        """
        super(STGNN, self).__init__()

        torch.set_default_device('cuda')

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.pred_length = pred_length
        self.use_graph_conv = use_graph_conv
        self.use_temp_conv = use_temp_conv
        self.relu = F.relu
        self.softmax = F.softmax
        self.mat_conv = nconv()
        self.adj_type = adj_type

        # Receptive field calculation
        kernel_size = 7
        self.receptive_field = int(1 + (kernel_size - 1) * (dilation_multiplier ** num_layers - 1) / (dilation_multiplier - 1)) \
            if dilation_multiplier > 1 \
            else num_layers * (kernel_size - 1) + 1
        # self.final_sequence_length = sequence_length - self.receptive_field + 1

        # Initialize dropout function
        # self.dropout = nn.Dropout(dropout_factor)
        self.dropout = dropout_factor

        # Initialize input convolution layer
        self.input_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=residual_chans,
            kernel_size=(1, 1))

        # Initialize graph convolution modules
        self.gc_layers = _init_layers(
            K=K,
            beta=beta,
            num_layers=num_layers,
            gc_support=gc_support_len,
            gc_order=gc_order,
            gc_dropout=gc_dropout,
            gc_c_in=conv_chans,
            gc_c_out=residual_chans,
            gc_nconv=self.mat_conv,
            use_diffusion=use_diffusion)

        # Intialize temporal convolution modules
        self.tc_layers = _init_layers(
            num_layers=num_layers,
            dilated_c_in=residual_chans,
            dilated_c_out=conv_chans,
            dilation_multiplier=dilation_multiplier)

        # Initialize skip convolution modules and
        # normalization layers
        self.skip_convs, self.layer_norms = _init_layers(
            seq_len=sequence_length,
            dilation_multiplier=dilation_multiplier,
            num_layers=num_layers,
            num_nodes=num_nodes,
            kernel_size=kernel_size,
            receptive_field=self.receptive_field,
            dilated_c_out=conv_chans,
            skip_chans=skip_chans,
            residual_chans=residual_chans)

        # Initialize starting skip convolution
        self.start_skip = nn.Conv2d(
            in_channels=c_in,
            out_channels=skip_chans,
            kernel_size=(1, sequence_length
                         if sequence_length > self.receptive_field
                         else self.receptive_field),
            bias=True)

        # Initialize ending skip convolution
        self.end_skip = nn.Conv2d(
            in_channels=residual_chans,
            out_channels=skip_chans,
            kernel_size=(1, sequence_length - self.receptive_field + 1
                         if sequence_length > self.receptive_field
                         else 1), bias=True)

        # Initialize the first output convolution layer
        self.output_conv_1 = nn.Conv2d(
            in_channels=skip_chans,
            out_channels=end_chans,
            kernel_size=(1, 1))

        # Initialize the second output convolution layer
        self.output_conv_2 = nn.Conv2d(
            in_channels=end_chans,
            out_channels=pred_length,
            kernel_size=(1, 1))

        # Node indices to be used in layer norms
        self.node_ind = torch.arange(num_nodes)

        self.mat_weights = []

        if self.adj_type == 'learned':
            for i in range(self.num_layers):
                # tup_l = (nn.Parameter(torch.randn((batch_size, num_nodes, num_nodes, sequence_length - 6 * (i + 1)))),
                #          nn.Parameter(torch.randn((batch_size, num_nodes, num_nodes, sequence_length - 6 * (i + 1)))))

                tup_l = (nn.Parameter(torch.randn((batch_size, num_nodes, num_nodes))),
                         nn.Parameter(torch.randn((batch_size, num_nodes, num_nodes))))
                self.mat_weights.append(tup_l)

    def forward(self, v):
        """
        Forward pass of the STGNN model.

        Parameters:
        - data (torch_geometric.data.Data): Input data containing node features and adjacency matrix.

        Returns:
        - torch.Tensor [batch_size, prediction_length, num_nodes]: Predicted output.
        """

        # Extract the feature tensor and adjacency matrix from data input.
        # x: torch.Tensor [batch_size, num_features, num_nodes, sequence_length]
        input = v
        # x = torch.transpose(x, 1, 2)
        # adj: [num_nodes, num_nodes]
        adj = None
        if len(self.mat_weights) <= 0:
            adj = [data.edge_attr for _ in range(self.num_layers)]

        elif len(self.mat_weights) > 0 and self.use_graph_conv:
            adj = [self.softmax(self.relu(torch.bmm(self.mat_weights[i][0], self.mat_weights[i][1].transpose(1, 2))), dim=-1)
                   for i in range(self.num_layers)]

        if self.sequence_length < self.receptive_field:
            input = F.pad(input, (self.receptive_field -
                          self.sequence_length, 0, 0, 0))

        x = self.input_conv(input)
        skip = self.start_skip(
            F.dropout(input, self.dropout, training=self.training))

        for i in range(self.num_layers):
            residual = x

            if self.use_temp_conv:
                x = self.tc_layers[i](x)
                x = F.dropout(x, self.dropout, training=self.training)

                s = x
                s = self.skip_convs[i](s)
                skip = s + skip

            if self.use_graph_conv:
                adj_i = adj[i]
                if not self.use_temp_conv:
                    if x.shape[1] != self.gc_layers[i].mh_1.mlp.mlp.in_channels // self.gc_layers[i].mh_1.K:
                        adjust_conv = nn.Conv2d(
                            x.shape[1], self.gc_layers[i].mh_1.mlp.mlp.in_channels // self.gc_layers[i].mh_1.K, kernel_size=(1, 1)).to('cuda')
                        x = adjust_conv(x)
                x = self.gc_layers[i](x, adj_i)

            if self.use_graph_conv and self.use_temp_conv:
                x = x + residual[:, :, :, -x.size(3):]
            else:
                x = residual[:, :, :, -x.size(3):]

            if self.use_graph_conv and self.use_temp_conv:
                x = self.layer_norms[i](x, self.node_ind)

        skip = self.end_skip(x) + skip
        x = F.relu(skip)
        x = F.relu(self.output_conv_1(x))
        x = self.output_conv_2(x)

        if self.use_graph_conv and not self.use_temp_conv:
            x = x.permute(0, 2, 1, 3)

            x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3))

            x = x[:, :, :self.pred_length]
        else:
            x = x.squeeze(3)

        return x


def construct_STGNN(
        data: Data,
        K: int = 10,
        beta: float = 0.05,
        residual_chans: int = 32,
        conv_chans: int = 32,
        skip_chans: int = 64,
        end_chans: int = 128,
        num_layers: int = 10,
        gc_support: int = 3,
        gc_order: int = 2,
        gc_dropout: float = 0.2,
        dilation_multiplier: int = 1,
        dropout_factor: float = 0.2,
        use_graph_conv: bool = True,
        use_temp_conv: bool = True,
        use_diffusion: bool = False,
        adj_type: str = 'learned') -> STGNN:

    batch_size, c_in, num_nodes, sequence_length = data.x.shape
    _, pred_length, _ = data.y.shape

    return STGNN(
        c_in=c_in,
        K=K,
        beta=beta,
        num_nodes=num_nodes,
        sequence_length=sequence_length,
        residual_chans=residual_chans,
        conv_chans=conv_chans,
        skip_chans=skip_chans,
        end_chans=end_chans,
        num_layers=num_layers,
        gc_support_len=gc_support,
        gc_dropout=gc_dropout,
        gc_order=gc_order,
        dilation_multiplier=dilation_multiplier,
        pred_length=pred_length,
        dropout_factor=dropout_factor,
        use_graph_conv=use_graph_conv,
        use_temp_conv=use_temp_conv,
        batch_size=batch_size,
        use_diffusion=use_diffusion,
        adj_type=adj_type)
