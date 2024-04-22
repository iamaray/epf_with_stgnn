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


class Output(nn.Module):
    def __init__(self):
        super(Output, self).__init__()
        pass


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
