from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.nn import init
import numbers
import os
import sys
# from utils.general_utils import compute_a_tilde

# sys.path.insert(0, os.path.abspath(".."))
import utils.general_utils as gutils


class SingleMixHop(nn.Module):
    def __init__(self, K, beta, adj):
        super(SingleMixHop, self).__init__()
        self.K = K
        self.beta = beta
        self.adj = gutils.compute_a_tilde(adj)
        self.W = self._init_weights()
        self._init_weights()

    def _init_weights(self):
        weights = []
        for k in range(self.K):
            w_k = nn.Parameter(torch.randn(data.x.shape[0], data.x.shape[1]))
            w_k.requires_grad = True
            weights.append(w_k)
        return torch.stack(weights)

    def _prop_step(self, past_hops):
        h_in = past_hops[-1]  # output of last pass
        h_0 = h_in
        curr_hops = [h_0]
        for k in range(1, self.K):
            curr_hops.append(
                torch.Tensor(self.beta * h_in + (1 - self.beta) * curr_hops[k-1]))

        return torch.stack(curr_hops)

    def _selection_step(self, curr_hops):
        h_out = torch.einsum('bik,bik->bik', curr_hops, self.W).sum(dim=0)
        return h_out

    def forward(self, past_hops):
        curr_hops = self._prop_step(past_hops)
        # curr_hops = torch.Tensor(curr_hops)
        h_out = self._selection_step(curr_hops)

        return h_out


class GraphConvolution(nn.Module):
    """
        The Graph Convolution (GC) module is composed of two mix-hop propagation
        layers, one taking A and the other taking A^T. A mix-hop propagation layer
        performs a horizontal mix-hop operation to compute the successive hidden features
        H^(k). It then feeds each of these into an MLP. The resultant output is then
        aggregated via a weighted sum.
    """

    def __init__(self, K, beta, adj):
        super(GraphConvolution, self).__init__()

        self.mh_1 = SingleMixHop(K, beta, adj)
        self.mh_2 = SingleMixHop(K, beta, torch.transpose(adj, 0, 1))

    def forward(self, past_hops):
        out_1 = self.mh_1(past_hops)
        out_2 = self.mh_2(past_hops)
        return torch.stack([out_1, out_2]).sum(dim=0)


class TemporalConvolution(nn.Module):
    """
        The Temporal Convolution (TC) module performs inception convolution:
            a concatenation of a series of 1-D convolution operations
            given by equation (12) of Wu et. al.
        And dilated convolution:
            a summation given by equation (13) of Wu et. al.
    """

    def __init__(self, in_features, out_features, dilation_factor=2):
        super(TemporalConvolution, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]  # different kernel sizes
        # divide the total number of output channels to match with each kernel size
        out_features = int(out_features/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(
                nn.Conv2d(in_features, out_features, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class Output(nn.Module):
    """
        A class that performs a 1D convolution to transform the input tensor.
    """

    def __init__(self, in_channels, out_channels):
        super(Output, self).__init__()
        # Use the provided input and output dimensions
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # [batch_size, in_channels, length] -> [batch_size, length, in_channels]
        x = x.permute(2, 1)
        x = self.conv1(x)
        # [batch_size, new_length, in_channels] -> [batch_size, in_channels, new_length]
        x = x.permute(2, 1)

        return x


class LayerNorm(nn.Module):  # performs layer normalization
    __constants__ = ['normalized_shape', 'weight',
                     'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
