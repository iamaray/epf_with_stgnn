import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
from __future__ import division
from torch.nn import init
import numbers

"""
    nconv, dy_nconv, and linear taken from 
    https://github.com/nnzhan/MTGNN/blob/master/layer.py
"""


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        """
        Summary: performs graph convolution on x via adjacency matrix A

        Args:
            x (Tensor): input tensor with shape (batch_size, num_channels, num_nodes, seq_len)
            A (Tensor): adjacency matrix

        Returns:
            x: contiguously-stored output tensor
        """
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        """
        Summary: performs dynamic graph convolution on x via adjacency matrix A.
                 I.e., a graph convolution subject to an adjacency matrix that
                 could change across iterations.
        Args:
            x (Tensor): input tensor with shape (batch_size, num_channels, num_nodes, seq_len)
            A (Tensor): adjacency matrix

        Returns:
            x: contiguously-stored output tensor
        """
        x = torch.einsum('ncvl,nvwl->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):  # acts as the MLP in the mix-hop layers
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


"""
    The Graph Convolution (GC) module is composed of two mix-hop propagation 
    layers, one taking A and the other taking A^T. A mix-hop propagation layer 
    performs a horizontal mix-hop operation to compute the successive hidden features
    H^(k). It then feeds each of these into an MLP. The resultant output is then
    aggregated via a weighted sum.
"""


class GraphConvolution(nn.Module):
    def __init__(self):
        super(GraphConvolution, self).__init__()
        pass


"""
    The Temporal Convolution (TC) module performs inception convolution:
        a concatenation of a series of 1-D convolution operations
        given by equation (12) of Wu et. al.
    And dilated convolution:
        a summation given by equation (13) of Wu et. al.

"""


class TemporalConvolution(nn.Module):
    def __init__(self):
        super(TemporalConvolution, self).__init__()
        pass


"""
    Details needed.
"""


class SkipConnection(nn.Module):
    def __init__(self):
        super(SkipConnection, self).__init__()
        pass


"""
    Details needed.
"""


class Output(nn.Module):
    def __init__(self):
        super(Output, self).__init__()
        pass
