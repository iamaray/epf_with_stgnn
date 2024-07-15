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


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # Adjusted einsum to handle batch dimension in A
        x = torch.einsum('ncvl,nvw->ncvl', (x, A))
        return x.contiguous()


class MixPropMLP(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(MixPropMLP, self).__init__()

        self.mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class SingleMixHop(nn.Module):
    def __init__(
            self,
            K,
            beta,
            c_in,
            c_out,
            nconv=nconv()):
        super(SingleMixHop, self).__init__()
        self.K = K
        self.beta = beta
        self.nconv = nconv
        self.mlp = MixPropMLP(c_in * K, c_out).to('cuda')

    def _init_weights(self, weight_shape):
        weights = []
        for k in range(self.K):
            w_k = nn.Parameter(torch.randn(weight_shape)).to('cuda')
            w_k.requires_grad = True
            weights.append(w_k)
        return torch.stack(weights)

    def _prop_step(self, x, adj, mlp=None):
        h_in = x.to('cuda')
        curr_hops = [h_in]
        for k in range(1, self.K):
            curr_hops.append(
                self.beta * h_in + (1 - self.beta)
                * self.nconv(curr_hops[k-1], adj))

        return torch.cat(curr_hops, dim=1)

    def _selection_step(self, curr_hops):
        # h_out = torch.einsum('awbik,awbik->awbik', curr_hops, self.W).sum(dim=0)
        h_out = self.mlp(curr_hops)
        return h_out

    def forward(self, x, adj):
        adj = compute_a_tilde(adj).to('cuda')
        curr_hops = self._prop_step(x, adj, self.mlp)
        h_out = self._selection_step(curr_hops)

        return h_out


class GraphConvolution(nn.Module):
    def __init__(self, K, beta, c_in, c_out, nconv):
        super(GraphConvolution, self).__init__()

        self.mh_1 = SingleMixHop(K, beta, c_in, c_out, nconv)
        self.mh_2 = SingleMixHop(K, beta, c_in, c_out, nconv)

    def forward(self, x, adj):
        out_1 = self.mh_1(x, adj).to('cuda')
        out_2 = self.mh_2(x, torch.transpose(adj, 1, 2)).to('cuda')

        return out_1 + out_2


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
