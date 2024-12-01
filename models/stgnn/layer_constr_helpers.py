import torch
import torch.nn as nn
from typing import Optional, Tuple
from .modules import *

import torch.nn as nn
from typing import Optional, Tuple


def _init_layers(
        K: Optional[int] = 0,
        beta: Optional[float] = 0.05,
        num_layers: int = 0,
        dilated_c_in: Optional[int] = None,
        dilated_c_out: Optional[int] = None,
        gc_c_in: Optional[int] = None,
        gc_c_out: Optional[int] = None,
        gc_dropout: Optional[float] = None,
        gc_support: Optional[int] = None,
        gc_order: Optional[int] = None,
        dilation_multiplier: int = 2,
        seq_len: Optional[int] = None,
        num_nodes: Optional[int] = None,
        kernel_size: Optional[int] = None,
        receptive_field: Optional[int] = None,
        skip_chans: Optional[int] = None,
        residual_chans: Optional[int] = None,
        gc_nconv: Optional[int] = None,
        use_diffusion: Optional[bool] = True) -> Tuple[nn.ModuleList, nn.ModuleList]:
    """
      Initializes layers based on the provided parameters.

      Parameters:
      - num_layers: Number of layers to initialize.
      - K, beta, gc_c_in, gc_c_out: Parameters for Graph Convolution layers.
      - dilated_c_in, dilated_c_out, dilation_multiplier: Parameters for Temporal Convolution layers.
      - seq_len, num_nodes, kernel_size, receptive_field, skip_chans, residual_chans: Parameters for Skip Convolution and Layer Normalization layers.

      Returns:
      - A tuple of nn.ModuleList instances for the initialized layers.
    """
    if num_layers > 0:
        gc_params_check = all(
            [gc_c_in, gc_c_out, gc_dropout, gc_support, gc_order])
        tc_params_check = all(
            [dilated_c_in, dilated_c_out, dilation_multiplier])
        skip_conv_layer_norm_check = all([dilation_multiplier, seq_len, num_nodes,
                                         kernel_size, receptive_field, skip_chans, residual_chans, dilated_c_out])

        if gc_params_check:
            return _init_gc_layers(
                num_layers,
                gc_c_in=gc_c_in,
                gc_c_out=gc_c_out,
                gc_nconv=gc_nconv,
                gc_support=gc_support,
                gc_dropout=gc_dropout,
                gc_order=gc_order,
                use_diffusion=use_diffusion,
                K=K,
                beta=beta)

        elif tc_params_check:
            return _init_tc_layers(
                num_layers,
                dilated_c_in,
                dilated_c_out,
                dilation_multiplier)

        elif skip_conv_layer_norm_check:
            return _init_skip_conv_layer_norm(
                seq_len,
                dilation_multiplier,
                num_layers,
                num_nodes,
                kernel_size,
                receptive_field,
                dilated_c_out,
                skip_chans,
                residual_chans)
        else:
            raise ValueError(
                "Given params not valid for any layer initializations")


def _init_gc_layers(
        num_layers: int,
        gc_support: int,
        gc_order: int,
        gc_dropout: float,
        gc_c_in: int,
        gc_c_out: int,
        K: int,
        beta: float,
        gc_nconv: None,
        use_diffusion: bool) -> nn.ModuleList:
    """
    Initializes Graph Convolution layers.

    Parameters:
    - num_layers: Number of layers to initialize.
    - K, beta, gc_c_in, gc_c_out: Parameters for the Graph Convolution layers.

    Returns:
    - nn.ModuleList of initialized Graph Convolution layers.
    """
    layers = nn.ModuleList()
    if use_diffusion:
        for _ in range(num_layers):
            layers.append(
                DiffusionGraphConvolution(
                    c_in=gc_c_in,
                    c_out=gc_c_out,
                    nconv=gc_nconv,
                    support_len=gc_support,
                    order=gc_order,
                    dropout=gc_dropout))
    else:
        for _ in range(num_layers):
            layers.append(
                GraphConvolution(
                    c_in=gc_c_in,
                    c_out=gc_c_out,
                    K=K,
                    beta=beta,
                    nconv=gc_nconv))

    return layers


def _init_tc_layers(
        num_layers: int,
        cin: int,
        cout: int,
        dilation_multiplier: int) -> nn.ModuleList:
    """
    Initializes Temporal Convolution layers.

    Parameters:
    - num_layers: Number of layers to initialize.
    - cin, cout: Input and output channel sizes for the layers.
    - dilation_multiplier: Dilation multiplier for the layers.

    Returns:
    - nn.ModuleList of initialized Temporal Convolution layers.
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
        residual_chans: int) -> Tuple[nn.ModuleList, nn.ModuleList]:
    """
    Initializes Skip Convolution and Layer Normalization layers.

    Parameters:
    - seq_len, dilation_multiplier, num_layers, num_nodes, kernel_size, receptive_field, conv_chans, skip_chans, residual_chans: Parameters for the layers.

    Returns:
    - A tuple of nn.ModuleList instances for the initialized Skip Convolution and Layer Normalization layers.
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
                    kernel_size=(1, max(seq_len - rf_size_j + 1, receptive_field - rf_size_j + 1))))

            layer_norms.append(
                LayerNorm((residual_chans, num_nodes, max(seq_len - rf_size_j + 1, receptive_field - rf_size_j + 1)), elementwise_affine=True))

    return skip_layers, layer_norms
