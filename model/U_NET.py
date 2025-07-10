
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from constant_autoregression.util import Printer, initialize_weights_xavier_uniform
import math


class ConvBlock_1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu") -> None:
        super().__init__()
        self.activation = activation
        #self.activation = ACTIVATION_REGISTRY.get(activation, None)
        #if self.activation is None:
            #raise NotImplementedError(f"Activation {activation} not implemented")

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        if norm:
            # Original used BatchNorm2d
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.activation(self.norm1(self.conv1(x)))
        h = self.activation(self.norm2(self.conv2(h)))
        return h


class Down_1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation=nn.GELU) -> None:
        super().__init__()
        self.conv = ConvBlock_1D(in_channels, out_channels, num_groups, norm, activation)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor):
        h = self.pool(x)
        h = self.conv(h)
        return h


class Up_1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation=nn.GELU) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock_1D(in_channels, out_channels, num_groups, norm, activation)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        #print("x1 ->", x1.shape)
        h = self.up(x1)
        #print("up(x1) ->", h.shape)

        h = torch.cat([x2, h], dim=1)
        #print("cat(x2,h) ->", h.shape)

        h = self.conv(h)
        #print("conv(h) ->", h.shape)
        return h


class U_NET_1D(nn.Module):
    """Our interpretation of the original U-Net architecture.

    Uses [torch.nn.GroupNorm][] instead of [torch.nn.BatchNorm2d][]. Also there is no `BottleNeck` block.

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input.
        time_future (int): Number of time steps in the output.
        hidden_channels (int): Number of channels in the hidden layers.
        activation (str): Activation function to use. One of ["gelu", "relu", "silu"].
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        padding: int,
        activation=nn.GELU(),
        
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels
        self.padding = padding
        #self.activation = ACTIVATION_REGISTRY.get(activation, None)
        # if self.activation is None:
        #     raise NotImplementedError(f"Activation {activation} not implemented")

        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 2)
        n_channels = hidden_channels
        self.image_proj = ConvBlock_1D(insize, n_channels, activation=activation)

        self.down = nn.ModuleList(
            [
                Down_1D(n_channels, n_channels * 2, activation=activation),
                Down_1D(n_channels * 2, n_channels * 4, activation=activation),
                Down_1D(n_channels * 4, n_channels * 8, activation=activation),
                Down_1D(n_channels * 8, n_channels * 16, activation=activation),
            ]
        )
        self.up = nn.ModuleList(
            [
                Up_1D(n_channels * 16, n_channels * 8, activation=activation),
                Up_1D(n_channels * 8, n_channels * 4, activation=activation),
                Up_1D(n_channels * 4, n_channels * 2, activation=activation),
                Up_1D(n_channels * 2, n_channels, activation=activation),
            ]
        )
        out_channels = time_future * (self.n_output_scalar_components + self.n_output_vector_components * 2)
        # should there be a final norm too? but we aren't doing "prenorm" in the original
        self.final = nn.Conv1d(n_channels, out_channels, kernel_size=(3), padding=(1))

    def forward(self, x):
        #assert x.dim() == 5
        x = F.pad(x.permute(0,2,1), [self.padding, self.padding ]).permute(0,2,1)
        #print("x_pad ->", x.shape)
        orig_shape = x.shape
        x = x.permute(0,2,1)
        #print(x.shape)
        h = self.image_proj(x)

        x1 = self.down[0](h)
        x2 = self.down[1](x1)
        x3 = self.down[2](x2)
        x4 = self.down[3](x3)

        #print("x4, x3 ->", x4.shape, x3.shape)
        x = self.up[0](x4, x3)

        #print("x, x2 ->", x.shape, x2.shape)
        x = self.up[1](x, x2)

        #print("x, x1 ->", x.shape, x1.shape)
        x = self.up[2](x, x1)

        #print("x, h ->", x.shape, h.shape)
        x = self.up[3](x, h)

        x = x[...,self.padding: -self.padding]

        #print("x ->", x.shape)
        x = self.final(x)
        #print("x ->", x.shape)

        x = x.permute(0,2,1)
        # x = x.reshape(
        #     orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:])
        #print("x ->", x.shape)
        return x