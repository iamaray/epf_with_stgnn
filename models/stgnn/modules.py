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
import utils.general_utils as gutils


class nconv(nn.Module):
    """
    Applies a normalized graph convolution operation.

    Methods:
        forward(x, A): Performs the graph convolution on input tensor x using adjacency matrix A.
    """

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nvw->ncvl', (x, A))
        return x.contiguous()


class MixPropMLP(nn.Module):
    """
    Applies a multi-layer perceptron with a single convolutional layer.

    Parameters:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        bias (bool): If True, adds a learnable bias to the output.

    Methods:
        forward(x): Applies the MLP on input tensor x.
    """

    def __init__(self, c_in, c_out, bias=True):
        super(MixPropMLP, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class SingleMixHop(nn.Module):
    """
    Implements a single layer of MixHop, a generalized graph convolution layer.

    Parameters:
        K (int): Number of propagation steps.
        beta (float): Mixing parameter.
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        nconv (nn.Module): Normalized convolution module.

    Methods:
        forward(x, adj): Applies the MixHop operation on input tensor x using adjacency matrix adj.
    """

    def __init__(self, K, beta, c_in, c_out, nconv=nconv()):
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

    def _prop_step(self, x, adj):
        h_in = x.to('cuda')
        curr_hops = [h_in]
        for k in range(1, self.K):
            curr_hops.append(self.beta * h_in + (1 - self.beta)
                             * self.nconv(curr_hops[k-1], adj))
        return torch.cat(curr_hops, dim=1)

    def _selection_step(self, curr_hops):
        h_out = self.mlp(curr_hops)
        return h_out

    def forward(self, x, adj):
        adj = gutils.compute_a_tilde(adj).to('cuda')
        curr_hops = self._prop_step(x, adj)
        h_out = self._selection_step(curr_hops)
        return h_out


class GraphConvolution(nn.Module):
    """
    Implements a graph convolution layer with two MixHop layers.

    Parameters:
        K (int): Number of propagation steps.
        beta (float): Mixing parameter.
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        nconv (nn.Module): Normalized convolution module.

    Methods:
        forward(x, adj): Applies the graph convolution on input tensor x using adjacency matrix adj.
    """

    def __init__(self, K, beta, c_in, c_out, nconv):
        super(GraphConvolution, self).__init__()
        self.mh_1 = SingleMixHop(K, beta, c_in, c_out, nconv)
        self.mh_2 = SingleMixHop(K, beta, c_in, c_out, nconv)

    def forward(self, x, adj):
        out_1 = self.mh_1(x, adj).to('cuda')
        out_2 = self.mh_2(x, torch.transpose(adj, 1, 2)).to('cuda')
        return out_1 + out_2


class DilatedInception(nn.Module):
    """
    Implements a dilated inception layer.

    Parameters:
        cin (int): Number of input channels.
        cout (int): Number of output channels.
        dilation_factor (int): Dilation factor for convolution layers.

    Methods:
        forward(input): Applies the dilated inception on input tensor.
    """

    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern),
                              dilation=(1, dilation_factor)).to('cuda'))

    def forward(self, input):
        x = [conv(input) for conv in self.tconv]
        x = [x_i[..., -x[-1].size(3):] for x_i in x]
        x = torch.cat(x, dim=1)
        return x


class TemporalConvolution(nn.Module):
    """
    Implements a temporal convolution layer with gated mechanisms.

    Parameters:
        cin (int): Number of input channels.
        cout (int): Number of output channels.
        dilation_factor (int): Dilation factor for convolution layers.

    Methods:
        forward(input): Applies the temporal convolution on input tensor.
    """

    def __init__(self, cin, cout, dilation_factor=2):
        super(TemporalConvolution, self).__init__()
        self.dilated_inception_1 = DilatedInception(cin, cout, dilation_factor)
        self.dilated_inception_2 = DilatedInception(cin, cout, dilation_factor)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        gate = self.relu(self.dilated_inception_1(input))
        filter = self.tanh(self.dilated_inception_2(input))
        res = gate * filter
        return res


class LayerNorm(nn.Module):
    """
    Applies Layer Normalization over a mini-batch of inputs.

    Parameters:
        normalized_shape (int or tuple): Input shape from an expected input.
        eps (float): A value added to the denominator for numerical stability.
        elementwise_affine (bool): If True, this module has learnable per-element affine parameters.

    Methods:
        forward(input, idx): Applies layer normalization on input tensor.
        reset_parameters(): Resets the parameters of the layer.
        extra_repr(): Returns a string representation of the layer.
    """
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
            self.weight = nn.Parameter(
                torch.Tensor(*normalized_shape)).to('cuda')
            self.bias = nn.Parameter(
                torch.Tensor(*normalized_shape)).to('cuda')
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)
