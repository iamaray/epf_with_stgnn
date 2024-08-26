import torch
import torch.nn as nn
from typing import Optional, Tuple
from modules import *


def _init_layers(
    num_layers: int = 0,
    K: Optional[int] = None,
    beta: Optional[float] = None,
    dilated_c_in: Optional[int] = None,
    dilated_c_out: Optional[int] = None,
    gc_c_in: Optional[int] = None,
    gc_c_out: Optional[int] = None,
    dilation_multiplier: int = 2,
    seq_len: Optional[int] = None,
    num_nodes: Optional[int] = None,
    kernel_size: Optional[int] = None,
    receptive_field: Optional[int] = None,
    skip_chans: Optional[int] = None,
    residual_chans: Optional[int] = None,
    gc_nconv: Optional[int] = None
) -> Tuple[nn.ModuleList, nn.ModuleList]:
    """
    Initializes layers based on the provided parameters.

    Parameters:
        num_layers (int): Number of layers to initialize.
        K (Optional[int]): Parameter for Graph Convolution layers.
        beta (Optional[float]): Parameter for Graph Convolution layers.
        dilated_c_in (Optional[int]): Input channel size for Temporal Convolution layers.
        dilated_c_out (Optional[int]): Output channel size for Temporal Convolution layers.
        gc_c_in (Optional[int]): Input channel size for Graph Convolution layers.
        gc_c_out (Optional[int]): Output channel size for Graph Convolution layers.
        dilation_multiplier (int): Dilation multiplier for Temporal Convolution layers.
        seq_len (Optional[int]): Sequence length for Skip Convolution and Layer Normalization layers.
        num_nodes (Optional[int]): Number of nodes for Skip Convolution and Layer Normalization layers.
        kernel_size (Optional[int]): Kernel size for Skip Convolution layers.
        receptive_field (Optional[int]): Receptive field for Skip Convolution layers.
        skip_chans (Optional[int]): Skip channel size for Skip Convolution layers.
        residual_chans (Optional[int]): Residual channel size for Skip Convolution layers.
        gc_nconv (Optional[int]): Number of convolutional layers for Graph Convolution.

    Returns:
        Tuple[nn.ModuleList, nn.ModuleList]: A tuple of nn.ModuleList instances for the initialized layers.
    """
    if num_layers > 0:
        gc_params_check = all(
            [K, beta, gc_c_in, gc_c_out])

        tc_params_check = all(
            [dilated_c_in, dilated_c_out, dilation_multiplier])

        skip_conv_layer_norm_check = all(
            [dilation_multiplier, seq_len, num_nodes, kernel_size, receptive_field, skip_chans, residual_chans, dilated_c_out])

        if gc_params_check:
            return _init_gc_layers(num_layers, K, beta, gc_c_in, gc_c_out, gc_nconv=gc_nconv)
        elif tc_params_check:
            return _init_tc_layers(num_layers, dilated_c_in, dilated_c_out, dilation_multiplier)
        elif skip_conv_layer_norm_check:
            return _init_skip_conv_layer_norm(
                seq_len, dilation_multiplier, num_layers, num_nodes, kernel_size, receptive_field, dilated_c_out, skip_chans, residual_chans
            )
        else:
            raise ValueError(
                "Given parameters are not valid for any layer initializations")


def _init_gc_layers(
    num_layers: int,
    K: int,
    beta: float,
    gc_c_in: int,
    gc_c_out: int,
    gc_nconv
) -> nn.ModuleList:
    """
    Initializes Graph Convolution layers.

    Parameters:
        num_layers (int): Number of layers to initialize.
        K (int): Number of attention heads.
        beta (float): Attention score scaling factor.
        gc_c_in (int): Input channel size for Graph Convolution layers.
        gc_c_out (int): Output channel size for Graph Convolution layers.
        gc_nconv (nn.Module): Graph Convolution module.

    Returns:
        nn.ModuleList: A list of initialized Graph Convolution layers.
    """
    layers = nn.ModuleList()
    for _ in range(num_layers):
        layers.append(GraphConvolution(
            K, beta, gc_c_in, gc_c_out, nconv=gc_nconv))
    return layers


def _init_tc_layers(
    num_layers: int,
    cin: int,
    cout: int,
    dilation_multiplier: int
) -> nn.ModuleList:
    """
    Initializes Temporal Convolution layers.

    Parameters:
        num_layers (int): Number of layers to initialize.
        cin (int): Input channel size for Temporal Convolution layers.
        cout (int): Output channel size for Temporal Convolution layers.
        dilation_multiplier (int): Dilation multiplier for Temporal Convolution layers.

    Returns:
        nn.ModuleList: A list of initialized Temporal Convolution layers.
    """
    layers = nn.ModuleList()
    for i in range(num_layers):
        layers.append(TemporalConvolution(cin, cout, dilation_multiplier ** i))
    return layers


def _init_skip_conv_layer_norm(
    seq_len: int,
    dilation_multiplier: int,
    num_layers: int,
    num_nodes: int,
    kernel_size: int,
    receptive_field: int,
    conv_chans: int,
    skip_chans: int,
    residual_chans: int
) -> Tuple[nn.ModuleList, nn.ModuleList]:
    """
    Initializes Skip Convolution and Layer Normalization layers.

    Parameters:
        seq_len (int): Sequence length for Skip Convolution layers.
        dilation_multiplier (int): Dilation multiplier for Skip Convolution layers.
        num_layers (int): Number of layers to initialize.
        num_nodes (int): Number of nodes.
        kernel_size (int): Kernel size for Skip Convolution layers.
        receptive_field (int): Receptive field for Skip Convolution layers.
        conv_chans (int): Convolution channel size for Skip Convolution layers.
        skip_chans (int): Skip channel size for Skip Convolution layers.
        residual_chans (int): Residual channel size for Skip Convolution layers.

    Returns:
        Tuple[nn.ModuleList, nn.ModuleList]: A tuple of nn.ModuleList instances for the initialized Skip Convolution and Layer Normalization layers.
    """
    skip_layers = nn.ModuleList()
    layer_norms = nn.ModuleList()

    for i in range(1):
        rf_size_i = int(1 + i * (kernel_size - 1) * (dilation_multiplier ** (num_layers - 1)) / (
            dilation_multiplier - 1)) if dilation_multiplier > 1 else i * num_layers * (kernel_size - 1) + 1

        for j in range(1, num_layers + 1):
            rf_size_j = int(rf_size_i + (kernel_size - 1) * (dilation_multiplier ** (j - 1)) / (
                dilation_multiplier - 1)) if dilation_multiplier > 1 else rf_size_i + j * (kernel_size - 1)

            skip_layers.append(
                nn.Conv2d(
                    in_channels=conv_chans,
                    out_channels=skip_chans,
                    kernel_size=(1, max(seq_len - rf_size_j + 1,
                                        receptive_field - rf_size_j + 1))
                ))

            layer_norms.append(
                LayerNorm(
                    (residual_chans, num_nodes,
                     max(seq_len - rf_size_j + 1, receptive_field - rf_size_j + 1)), elementwise_affine=True))

    return skip_layers, layer_norms
