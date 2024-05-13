import torch
import torch.nn as nn
import torch.nn.functional as F
import modules

def _init_layers(
    num_layers=0,
    K=None,
    beta=None,
    dilated_c_in=None,
    dilated_c_out=None,
    gc_c_in=None,
    gc_c_out=None,
    dilation_multiplier=2,
    seq_len=None,
    num_nodes=None,
    kernel_size=None,
    receptive_field=None,
    skip_chans=None,
    residual_chans=None):
    """
    Initializes different types of layers based on the specified parameters.

    Args:
        num_layers (int): Number of layers to initialize.
        K (int): Number of hops (neighborhoods) to consider in graph convolutions.
        beta (float): Mixing parameter for graph convolutions.
        dilated_c_in (int): Number of input channels for temporal convolution.
        dilated_c_out (int): Number of output channels for temporal convolution.
        dilation_multiplier (int): Factor to increase dilation for each layer of temporal convolution.
        seq_len (int): Length of the input sequences.
        num_nodes (int): Number of nodes in the graph.
        kernel_size (int): Size of the convolution kernel.
        receptive_field (int): Size of the receptive field.
        skip_chans (int): Number of output channels for skip connections.
        residual_chans (int): Number of residual channels.

    Returns:
        Union[nn.ModuleList, Tuple[nn.ModuleList, nn.ModuleList]]: Initialized layers as specified by the input parameters.
    """

    if num_layers > 0:
        gc_params_check = (K and beta and gc_c_in and gc_c_out)  # Check if all graph convolution parameters are set
        tc_params_check = (dilated_c_in and dilated_c_out and dilation_multiplier)  # Check for temporal convolution parameters
        skip_conv_layer_norm_check = (dilation_multiplier and seq_len and num_nodes \
                                      and kernel_size and receptive_field and skip_chans and residual_chans \
                                      and dilated_c_out)  # Check for skip convolution and layer normalization parameters

    # Dispatch initialization based on parameter sets provided
    if gc_params_check is not None:
        return _init_gc_layers(
            num_layers=num_layers,
            K=K,
            beta=beta,
            gc_c_in=gc_c_in,
            gc_c_out=gc_c_out)

    elif tc_params_check is not None:
        return _init_tc_layers(
            num_layers,
            dilated_c_in,
            dilated_c_out,
            dilation_multiplier)

    elif skip_conv_layer_norm_check is not None:
        return _init_skip_conv_layer_norm(
            num_layers=num_layers,
            seq_len=seq_len,
            dilation_multiplier=dilation_multiplier,
            num_nodes=num_nodes,
            kernel_size=kernel_size,
            receptive_field=receptive_field,
            conv_chans=dilated_c_out,
            skip_chans=skip_chans,
            residual_chans=residual_chans)


def _init_gc_layers(num_layers, K, beta, gc_c_in, gc_c_out):
    """
    Initializes a series of graph convolution layers.

    Args:
        num_layers (int): Number of graph convolution layers to initialize.
        K (int): Number of hops (neighborhoods) to consider.
        beta (float): Mixing parameter for combining features from different hops.
        gc_c_in (int): Number of input channels for each graph convolution layer.
        gc_c_out (int): Number of output channels for each graph convolution layer.

    Returns:
        nn.ModuleList: A list of initialized graph convolution layers.
    """
    layers = nn.ModuleList()

    for i in range(num_layers):
        gc_i = modules.GraphConvolution(K=K, beta=beta, c_in=gc_c_in, c_out=gc_c_out)  # Initialize each graph convolution layer
        layers.append(gc_i)

    return layers

def _init_tc_layers(num_layers, cin, cout, dilation_multiplier):
    """
    Initializes a series of temporal convolution layers with increasing dilation.

    Args:
        num_layers (int): Number of temporal convolution layers to initialize.
        cin (int): Number of input channels for each layer.
        cout (int): Number of output channels for each layer.
        dilation_multiplier (int): Factor to increase dilation for each subsequent layer.

    Returns:
        nn.ModuleList: A list of initialized temporal convolution layers.
    """
    layers = nn.ModuleList()

    for i in range(num_layers):
        tc_i = modules.TemporalConvolution(cin, cout, dilation_multiplier ** i)  # Exponentially increase dilation for each layer
        layers.append(tc_i)

    return layers

def _init_skip_conv_layer_norm(
    seq_len,
    dilation_multiplier,
    num_layers,
    num_nodes,
    kernel_size,
    receptive_field,
    conv_chans,
    skip_chans,
    residual_chans):
    """
    Initializes skip convolution layers and corresponding layer normalization layers.

    Args:
        seq_len (int): Length of the input sequences.
        dilation_multiplier (int): Dilation factor for convolutions.
        num_layers (int): Number of convolution layers.
        num_nodes (int): Number of nodes in the graph.
        kernel_size (int): Size of the convolution kernel.
        receptive_field (int): Size of the receptive field.
        conv_chans (int): Number of channels for convolution operations.
        skip_chans (int): Number of channels for skip connections.
        residual_chans (int): Number of channels for residuals.

    Returns:
        Tuple[nn.ModuleList, nn.ModuleList]: A tuple containing lists of skip convolution layers and layer normalization layers.
    """
    skip_layers = nn.ModuleList()
    layer_norms = nn.ModuleList()

    for i in range(1):  # Currently loops only once
        # Calculate the initial size of the receptive field for the i-th skip layer
        if dilation_multiplier > 1:
            rf_size_i = int(1 + i * (kernel_size - 1) * (dilation_multiplier ** num_layers-1) / (dilation_multiplier - 1))
        else:
            rf_size_i = i * num_layers * (kernel_size - 1) + 1

        new_dilation = 1
        for j in range(1, num_layers + 1):
            # Adjust receptive field size based on current layer and dilation
            if dilation_multiplier > 1:
                rf_size_j = int(rf_size_i + (kernel_size - 1) * (dilation_multiplier ** j-1)/(dilation_multiplier - 1))
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)

            # Define convolutional and normalization layers based on calculated receptive field size
            if seq_len > receptive_field:
                skip_layers.append(nn.Conv2d(
                    in_channels=conv_chans,
                    out_channels=skip_chans,
                    kernel_size=(1, seq_len - rf_size_j + 1)))  # Convolution adapted to the remaining sequence length
                layer_norms.append(modules.LayerNorm((
                    residual_chans,
                    num_nodes,
                    seq_len - rf_size_j + 1),
                    elementwise_affine=True))  # Normalization layer matches output of convolution
            else:
                skip_layers.append(nn.Conv2d(
                    in_channels=conv_chans,
                    out_channels=skip_chans,
                    kernel_size=(1, receptive_field - rf_size_j + 1)))  # Smaller convolution
                layer_norms.append(modules.LayerNorm((
                    residual_chans,
                    num_nodes,
                    receptive_field - rf_size_j + 1),
                    elementwise_affine=True))

        return skip_layers, layer_norms


class STGNN(nn.Module):
     """
     Spatio-Temporal Graph Neural Network model for handling graph-based time series data.

     Attributes:
         c_in (int): Number of input channels.
         residual_chans (int): Number of channels in residual connections.
         conv_chans (int): Number of channels in convolution layers.
         skip_chans (int): Number of channels in skip connections.
         end_chans (int): Number of output channels before prediction layer.
         num_layers (int): Total number of layers in the model.
         K (int): Number of hops for graph convolutions.
         beta (float): Mixing parameter for graph convolutions.
         dilation_multiplier (int): Dilation multiplier for temporal convolutions.
         pred_length (int): Length of the prediction output.
         dropout_factor (float): Dropout rate.
         num_nodes (int): Number of nodes in the graph.
         sequence_length (int): Length of input sequences.
         receptive_field (int): Calculated receptive field based on dilation and kernel size.
         final_sequence_length (int): Length of sequences after processing through the model.
     """
    def __init__(
            self,
            c_in=3,
            residual_chans=32,
            conv_chans=32,
            skip_chans=64,
            end_chans=128,
            num_layers=10,
            K=10,
            beta=0.05,
            dilation_multiplier=1,
            pred_length=24,
            dropout_factor=0.2,
            num_nodes=3,
            sequence_length=96):
        super(STGNN, self).__init__()

        # input of shape [batch_size, num_chans, num_features, seq_length]

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.receptive_field = None
        self.sequence_length = sequence_length
        self.pred_length = pred_length

        kernel_size = 7
        if dilation_multiplier > 1:
            self.receptive_field = int(
                1 + (kernel_size-1) * (dilation_multiplier**num_layers-1)/(dilation_multiplier-1))
        else:
            self.receptive_field = num_layers*(kernel_size-1) + 1

        self.final_sequence_length = self.sequence_length - self.receptive_field + 1

        # apply a dropout after each temporal convolution module
        self.dropout = nn.Dropout(dropout_factor)

        # input convolution
        self.input_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=residual_chans,
            kernel_size=(1,1))

        # graph convs
        self.gc_layers = _init_layers(
            num_layers=num_layers,
            K=K,
            beta=beta,
            gc_c_in=conv_chans,
            gc_c_out=residual_chans)
        print(self.gc_layers)

        # temporal convs
        self.tc_layers = _init_layers(
            num_layers=num_layers,
            dilated_c_in=residual_chans,
            dilated_c_out=conv_chans,
            dilation_multiplier=dilation_multiplier)
        print(self.tc_layers)

        # skip convs
        # layer norms
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

        self.start_skip = None
        self.end_skip = None

        if self.sequence_length > self.receptive_field:
            self.start_skip = nn.Conv2d(
                in_channels=c_in,
                out_channels=skip_chans,
                kernel_size=(1, self.sequence_length),
                bias=True)
            self.end_skip = nn.Conv2d(
                in_channels=residual_chans,
                out_channels=skip_chans,
                kernel_size=(1, self.sequence_length-self.receptive_field+1),
                bias=True)
        else:
            self.start_skip = nn.Conv2d(
                in_channels=c_in,
                out_channels=skip_chans,
                kernel_size=(1, self.receptive_field),
                bias=True)
            self.end_skip = nn.Conv2d(
                in_channels=residual_chans,
                out_channels=skip_chans, kernel_size=(1, 1),
                bias=True)


        self.output_conv_1 = nn.Conv2d(
            in_channels=skip_chans,
            out_channels=end_chans,
            kernel_size=(1, 1))

        self.output_conv_2 = nn.Conv2d(
            in_channels=end_chans,
            out_channels=self.pred_length,
            kernel_size=(1, 1))

        self.node_ind = torch.arange(self.num_nodes)

    def forward(self, data):
        x = data.x
        adj = data.edge_attr

        if self.sequence_length < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field-self.sequence_length, 0, 0, 0))


        skip = self.start_skip(self.dropout(x))
        x = self.input_conv(x)
        x = self.dropout(x)


        for i in range(self.num_layers):
            print(i)
            residual = x
            x = self.tc_layers[i](x)
            x = self.dropout(x)

            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            print('here: ', x.shape)

            x = self.gc_layers[i](x, adj)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.layer_norms[i](x, self.node_ind)

        skip = self.end_skip(x) + skip
        x = F.relu(skip)
        print('here2: ', x.shape)
        x = F.relu(self.output_conv_1(x))
        print('here3: ', x.shape)
        x = self.output_conv_2(x)

        return x