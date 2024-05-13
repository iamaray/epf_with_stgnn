import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numbers

"""
    The Graph Convolution (GC) module is composed of two mix-hop propagation 
    layers, one taking A and the other taking A^T. A mix-hop propagation layer 
    performs a horizontal mix-hop operation to compute the successive hidden features
    H^(k). It then feeds each of these into an MLP. The resultant output is then
    aggregated via a weighted sum.
"""

class nconv(nn.Module):
    """
    Applies a graph convolution by multiplying the input features with
    the adjacency matrix using Einstein summation notation. It simplifies feature propagation
    in graph neural networks.

    Attributes:
        None.
    """
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        """
        Forward pass for the neighborhood convolution.

        Args:
            x (Tensor): The input features of shape (batch_size, num_channels, num_nodes).
            A (Tensor): The adjacency matrix of shape (num_nodes, num_nodes).

        Returns:
            Tensor: The output features after applying graph convolution, maintaining input shape.
        """
        x = torch.einsum('bcl,vw->bcl', (x, A))  # Performs matrix multiplication using Einstein summation convention.
        return x.contiguous()  # Ensures memory is contiguous for any subsequent operations.

def compute_a_tilde(adj):
    """
    Computes the normalized adjacency matrix with added self-loops to help
    the flow of gradients during training.

    Args:
        adj (Tensor): The original adjacency matrix of shape (num_nodes, num_nodes).

    Returns:
        Tensor: The normalized adjacency matrix with self-loops.
    """
    rsum = torch.sum(adj, -1)  # Sum of each row of the adjacency matrix.
    d = 1 + rsum  # Degree matrix calculation with added self-loops.
    d_inv = torch.pow(d, -1)  # Inverse of the degree matrix.
    d_inv[torch.isinf(d_inv)] = 0.  # Handles potential division by zero.
    d_mat_inv = torch.diagflat(d_inv)  # Converts the inverse degrees to a diagonal matrix.
    adj_plus_I = adj + torch.eye(adj.shape[0])  # Adds the identity matrix to include self-loops.
    return torch.matmul(d_mat_inv, adj_plus_I)  # Normalizes adjacency matrix by the degree matrix.

class MixPropMLP(nn.Module):
    """
    Uses a single-layer convolution with kernel size (1,1) to transform each node's feature independently.

    Attributes:
        mlp (Conv2d): The 2D convolutional layer.
    """
    def __init__(self, c_in, c_out, bias=True):
        super(MixPropMLP, self).__init__()

        self.mlp = torch.nn.Conv2d(
            c_in,
            c_out,
            kernel_size=(1, 1),  # No spatial aggregation, acting like a per-node MLP
            padding=(0, 0),  # No padding.
            stride=(1, 1),
            bias=bias)  # Include bias

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the 2D convolution.
        """
        return self.mlp(x)

class SingleMixHop(nn.Module):
    """
    Implements a single mix-hop propagation step with an MLP at each hop.

    Perform successive hop operations and then combine
    the results using an MLP. Useful for creating multiple layers of
    graph convolutions that can capture different hop distances.

    Attributes:
        K (int): Number of hops to consider.
        beta (float): Mixing parameter to balance between hops.
        nconv (nconv): The neighborhood convolution module.
        mlp (MixPropMLP): Multi-layer perceptron to combine features from different hops.
    """
    def __init__(self, K, beta, c_in, c_out):
        super(SingleMixHop, self).__init__()
        self.K = K
        self.beta = beta
        self.nconv = nconv()
        self.mlp = MixPropMLP(c_in * K, c_out)

    def _init_weights(self, weight_shape):
        """
        Initializes weights for the mix-hop operation.

        Args:
            weight_shape (tuple): The shape of the weight tensor to be initialized.

        Returns:
            Tensor: A stacked tensor of initialized weights.
        """
        weights = []
        for k in range(self.K):
            w_k = nn.Parameter(torch.randn(weight_shape))
            w_k.requires_grad = True
            weights.append(w_k)
        return torch.stack(weights)

    def _prop_step(self, x, adj, mlp=None):
        """
        Propagates the input features through the graph for a specified number of hops.

        Args:
            x (Tensor): The initial node features.
            adj (Tensor): The adjacency matrix.
            mlp (MixPropMLP, optional): Optional MLP to process features after each hop.

        Returns:
            Tensor: Concatenated features from all hops.
        """
        h_in = x
        curr_hops = [h_in]  # Initializes list of hop results with input features.
        for k in range(1, self.K):
            # Aggregates results from each hop into the list.
            curr_hops.append(self.beta * h_in + (1 - self.beta) * self.nconv(curr_hops[k-1], adj))

        return torch.cat(curr_hops, dim=1)

    def _selection_step(self, curr_hops):
        """
        Applies an MLP to the concatenated hop features to select and combine them into a single output.

        Args:
            curr_hops (Tensor): Concatenated features from different hops.

        Returns:
            Tensor: The output features after MLP processing.
        """
        h_out = self.mlp(curr_hops)  # Applies MLP to combined hop features to produce output.
        return h_out

    def forward(self, x, adj):
        """
        Performs the full operation of mix-hop propagation followed by feature selection.

        Args:
            x (Tensor): Initial node features.
            adj (Tensor): The adjacency matrix to use for feature propagation.

        Returns:
            Tensor: The output features after mix-hop propagation and MLP processing.
        """
        adj = compute_a_tilde(adj)
        curr_hops = self._prop_step(x, adj, self.mlp)
        h_out = self._selection_step(curr_hops)

        return h_out

class GraphConvolution(nn.Module):
    """
    Combines two single mix-hop layers that process both the adjacency matrix and its transpose.

    Encapsulates the complete graph convolution operation with two pathways to ensure
    that information flow is captured in both directions along the edges.

    Attributes:
        mh_1 (SingleMixHop): The first mix-hop layer that processes the original adjacency matrix.
        mh_2 (SingleMixHop): The second mix-hop layer that processes the transpose of the adjacency matrix.
    """
    def __init__(self, K, beta, c_in, c_out):
        super(GraphConvolution, self).__init__()

        self.mh_1 = SingleMixHop(K, beta, c_in, c_out)
        self.mh_2 = SingleMixHop(K, beta, c_in, c_out)

    def forward(self, x, adj):
        """
        Forward pass of the graph convolution operation.

        Args:
            x (Tensor): Input node features.
            adj (Tensor): Adjacency matrix of the graph.

        Returns:
            Tensor: Output features after applying graph convolution on both the adjacency matrix and its transpose.
        """
        out_1 = self.mh_1(x, adj)
        out_2 = self.mh_2(x, torch.transpose(adj, 0, 1))

        return out_1 + out_2


"""
    The Temporal Convolution (TC) module performs inception convolution:
        a concatenation of a series of 1-D convolution operations
        given by equation (12) of Wu et. al.
    And dilated convolution:
        a summation given by equation (13) of Wu et. al.

"""


class DilatedInception(nn.Module):
    """
    Implements a series of dilated convolutions with different kernel sizes to capture temporal patterns.

    Perform dilated convolutions at various rates, which is useful
    for capturing features across different temporal resolutions.

    Attributes:
        tconv (ModuleList): List of 2D convolution layers with increasing dilation factors.
        kernel_set (list): Set of kernel sizes used in the convolutions.
    """
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1,kern), dilation=(1,dilation_factor)))
            # Creates a series of dilated convolutions, each with a different dilation rate
            # to capture wider temporal context.

    def forward(self,input):
        """
        Forward pass of the dilated inception module.

        Args:
            input (Tensor): Input tensor with dimensions (batch, channels, height, width).

        Returns:
            Tensor: Concatenated output from all dilated convolutions, ensuring same dimensionality at the end.
        """
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]

        x = torch.cat(x, dim=1)

        return x

class TemporalConvolution(nn.Module):
    """
    Combines the outputs of two dilated inception modules using a gating mechanism.

    Enables the model to learn more complex temporal features by applying
    a nonlinear combination of two separate pathways.

    Attributes:
        dilated_inception_1 (DilatedInception): First dilated inception for the gating signal.
        dilated_inception_2 (DilatedInception): Second dilated inception for the filtering signal.
        relu (ReLU): ReLU activation function for the gate.
        tanh (Tanh): Tanh activation function for the filter.
    """
    def __init__(self, cin, cout, dilation_factor=2):
        super(TemporalConvolution, self).__init__()
        self.dilated_inception_1 = DilatedInception(cin, cout, dilation_factor)
        self.dilated_inception_2 = DilatedInception(cin, cout, dilation_factor)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        """
        Forward pass of the temporal convolution module.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: The result of applying a gated mechanism between two dilated inception module outputs.
        """
        gate = self.relu(self.dilated_inception_1(input))  # Applies ReLU nonlinearity to gate features.
        filter = self.tanh(self.dilated_inception_2(input))  # Applies Tanh nonlinearity to filter features.
        res = gate * filter  # Element-wise multiplication of gate and filter to control information flow.
        return res


class LayerNorm(nn.Module):
    """
    Performs layer normalization over a mini-batch of inputs.

    Attributes:
        normalized_shape (tuple): The shape of the input tensor that will receive normalization.
        eps (float): A value added to the denominator for numerical stability.
        elementwise_affine (bool): A flag that enables learning of scaling and shifting parameters.
        weight (Tensor): The scale tensor, learned if `elementwise_affine` is True.
        bias (Tensor): The shift tensor, learned if `elementwise_affine` is True.
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
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize or reset parameters to their default values.
        """
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        """
        Forward pass of the layer normalization.

        Args:
            input (Tensor): Input tensor to be normalized.
            idx (int): Index specifying which slice of the parameters to use.

        Returns:
            Tensor: The normalized output tensor.
        """
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        """
        Set the extra representation of the module

        Returns:
            str: String showing the normalized shape and additional parameters.
        """
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
