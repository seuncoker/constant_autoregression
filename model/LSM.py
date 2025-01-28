"""
@author: Haixu Wu
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math


################################################################
# Multiscale modules 2D
################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConv_1d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_1d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv_1d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_1d(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv_1d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_1d(in_channels, out_channels)

    def forward(self, x1, x2):
        # print("up ....")

        # print("x1 ->", x1.shape)
        # print("x2 ->", x2.shape)

        x1 = self.up(x1)

        # print("x1 ->", x1.shape)
        # print("x2 ->", x2.shape)

        # input is CHW
        #diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[-1] - x1.size()[-1]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        #print("cat(x1,x2) ->", x.shape)
        x = self.conv(x)
        #print("conv(x) ->", x.shape)
        return x


class OutConv_1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class NeuralSpectralBlock1d(nn.Module):
    def __init__(self, width, num_basis, patch_size=[3], num_token=4):
        super(NeuralSpectralBlock1d, self).__init__()
        self.patch_size = patch_size
        self.width = width
        self.num_basis = num_basis

        # basis
        self.modes_list = (1.0 / float(num_basis)) * torch.tensor([i for i in range(num_basis)],
                                                                  dtype=torch.float).to(device)
        self.weights = nn.Parameter(
            (1 / (width)) * torch.rand(width, self.num_basis * 2, dtype=torch.float))
        # latent
        self.head = 8
        self.num_token = num_token
        self.latent = nn.Parameter(
            (1 / (width)) * torch.rand(self.head, self.num_token, width // self.head, dtype=torch.float))
        self.encoder_attn = nn.Conv1d(self.width, self.width * 2, kernel_size=1, stride=1)
        self.decoder_attn = nn.Conv1d(self.width, self.width, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        # q,k,v: B H L C/H
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def latent_encoder_attn(self, x):
        #print("latent encoding ..")
        # x: B C H W
        B, C, H = x.shape
        L = H
        #print("latent ->", self.latent.shape)
        latent_token = self.latent[None, :, :, :].repeat(B, 1, 1, 1)
        #print("latent_token ->", latent_token.shape)
        
        # print("x ->", x.shape)
        # x_tmp = self.encoder_attn(x)
        # print("x_tmp ->", x_tmp.shape)
        # x_tmp = x_tmp.view(B, C * 2, -1)
        # print("x_tmp ->", x_tmp.shape)
        # x_tmp = x_tmp.permute(0, 2, 1).contiguous()
        # print("x_tmp ->", x_tmp.shape)
        # x_tmp = x_tmp.view(B, L, self.head, C // self.head, 2)
        # print("x_tmp ->", x_tmp.shape)
        # x_tmp = x_tmp.permute(4, 0, 2, 1, 3).contiguous()
        # print("x_tmp ->", x_tmp.shape)

        # print(self.encoder_attn(x).shape)
        # print(self.encoder_attn(x).view(B, C * 2, -1).shape)
        
        x_tmp = self.encoder_attn(x).view(B, C * 2, -1).permute(0, 2, 1).contiguous() \
            .view(B, L, self.head, C // self.head, 2).permute(4, 0, 2, 1, 3).contiguous()
        
        #print("x_tmp ->", x_tmp.shape)
    

        #print("latent_token ->", self.self_attn(latent_token, x_tmp[0], x_tmp[1]).shape )
        latent_token = self.self_attn(latent_token, x_tmp[0], x_tmp[1]) + latent_token
        latent_token = latent_token.permute(0, 1, 3, 2).contiguous().view(B, C, self.num_token)
        #print("latent_token ->", latent_token.shape)
        return latent_token

    def latent_decoder_attn(self, x, latent_token):
        # x: B C L
        x_init = x
        B, C, H = x.shape
        L = H
        latent_token = latent_token.view(B, self.head, C // self.head, self.num_token).permute(0, 1, 3, 2).contiguous()
        x_tmp = self.decoder_attn(x).view(B, C, -1).permute(0, 2, 1).contiguous() \
            .view(B, L, self.head, C // self.head).permute(0, 2, 1, 3).contiguous()
        x = self.self_attn(x_tmp, latent_token, latent_token)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, H) + x_init  # B H L C/H
        return x

    def get_basis(self, x):
        # x: B C N
        x_sin = torch.sin(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        x_cos = torch.cos(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        return torch.cat([x_sin, x_cos], dim=-1)
    

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def forward(self, x):
        B, C, H = x.shape
        #print("processing ..")

        #print("x ->", x.shape)
        # x_inter = x.view(x.shape[0], x.shape[1], self.patch_size[0], x.shape[-1] // self.patch_size[0])
        # print("x_inter ->", x_inter)


        # x = x.view(x.shape[0], x.shape[1],
        #            x.shape[2] // self.patch_size[0], self.patch_size[0], x.shape[3] // self.patch_size[1],
        #            self.patch_size[1]).contiguous() \
        #     .permute(0, 2, 4, 1, 3, 5).contiguous() \
        #     .view(x.shape[0] * (x.shape[2] // self.patch_size[0]) * (x.shape[3] // self.patch_size[1]), x.shape[1],
        #           self.patch_size[0],
        # #           self.patch_size[1])
        # print(x.view(x.shape[0], x.shape[1],
        #     x.shape[2] // self.patch_size[0], self.patch_size[0]).contiguous().shape)
        
        x = x.view(x.shape[0], x.shape[1],
            x.shape[2] // self.patch_size[0], self.patch_size[0]).contiguous() \
            .permute(0, 2, 1, 3).contiguous() \
            .view(x.shape[0]*(x.shape[2] // self.patch_size[0]),  x.shape[1], self.patch_size[0] )
        
        #print("x ->",x.shape)
        
        # Neural Spectral

        # (1) encoder
        latent_token = self.latent_encoder_attn(x)

        # (2) transition
        latent_token_modes = self.get_basis(latent_token)
        #print("latent_token_modes ->", latent_token_modes.shape)
        #print("self.weights ->", self.weights.shape)
        latent_token = self.compl_mul2d(latent_token_modes, self.weights) + latent_token
        #print("latent_token ->", latent_token.shape)


        # (3) decoder
        x = self.latent_decoder_attn(x, latent_token)

        #print("x ->", x.shape)

        # de-patchify
        x = x.view(B, (H // self.patch_size[0]), C, self.patch_size[0]).permute(0, 2, 1, 3).contiguous() \
            .view(B, C, H).contiguous()
        
        #print("x ->", x.shape)
        return x



class LSM_1D(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, num_token, num_basis, patch_size, padding, bilinear=False):
        super(LSM_1D, self).__init__()
        in_channels = in_dim
        out_channels = out_dim
        width = d_model
        num_token = num_token
        num_basis = num_basis
        patch_size = [int(patch_size)]
        padding = [int(padding)]
        # multiscale modules
        self.inc = DoubleConv_1d(width, width)
        self.down1 = Down_1d(width, width * 2)
        self.down2 = Down_1d(width * 2, width * 4)
        self.down3 = Down_1d(width * 4, width * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down_1d(width * 8, width * 16 // factor)
        self.up1 = Up_1d(width * 16, width * 8 // factor, bilinear)
        self.up2 = Up_1d(width * 8, width * 4 // factor, bilinear)
        self.up3 = Up_1d(width * 4, width * 2 // factor, bilinear)
        self.up4 = Up_1d(width * 2, width, bilinear)
        self.outc = OutConv_1d(width, width)
        # Patchified Neural Spectral Blocks
        self.process1 = NeuralSpectralBlock1d(width, num_basis, patch_size, num_token)
        self.process2 = NeuralSpectralBlock1d(width * 2, num_basis, patch_size, num_token)
        self.process3 = NeuralSpectralBlock1d(width * 4, num_basis, patch_size, num_token)
        self.process4 = NeuralSpectralBlock1d(width * 8, num_basis, patch_size, num_token)
        self.process5 = NeuralSpectralBlock1d(width * 16 // factor, num_basis, patch_size, num_token)
        # projectors
        self.padding = padding
        self.fc0 = nn.Linear(in_channels + 1, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        #print("x ->", x.shape)
        x = self.fc0(x)
        #print("x ->", x.shape)

        x = x.permute(0, 2, 1)
        #print("x ->", x.shape)

        if not all(item == 0 for item in self.padding):
            #print(self.padding)
            x = F.pad(x, [self.padding[0], self.padding[0] ])

        #print("x_pad ->", x.shape )
        x1 = self.inc(x)

        #print("x1 ->", x1.shape)
        
        x2 = self.down1(x1)
        #print("x2 ->", x2.shape)

        x3 = self.down2(x2)
        #print("x3 ->", x3.shape)

        x4 = self.down3(x3)
        #print("x4 ->", x4.shape)
        
        x5 = self.down4(x4)
        #print("x5 ->", x5.shape)

        # print("processing x4")
        # x4_p = self.process4(x4)

        # print("processing x5")
        # x5_p = self.process5(x5)

        # x = self.up1(x5_p, x4_p)
        
        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        x = self.outc(x)

        if not all(item == 0 for item in self.padding):
            x = x[..., self.padding[0]:-self.padding[0]]
        x = x.permute(0, 2, 1)
        #print("x ->", x.shape)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        #print("x ->", x.shape)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        #gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        #gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return gridx.to(device)