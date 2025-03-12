import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from constant_autoregression.util import Printer, initialize_weights_xavier_uniform
import math





p = Printer(n_digits=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Normalizer_1D(nn.Module):
    """
    Normalizer class for data preprocessing.
    """

    def __init__(self, eps=1e-5):
        """
        Initializes the Normalizer class.

        Args:
            eps: A small value to avoid division by zero (default: 1e-5).
        """
        super(Normalizer_1D, self).__init__()
        # self.register_buffer("running_mean", torch.zeros(1))
        # self.register_buffer("running_std", torch.ones(1))

        self.running_mean = torch.zeros(1).to(device)
        self.running_std = torch.ones(1).to(device)
        self.eps = eps

    def forward(self, x):
        """
        Normalizes the input tensor by subtracting the mean and dividing by the standard deviation.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        B,D,I = x.shape

        x_pool_dim = x.view(-1, x.shape[-1])

        # if not self.training:
        #     mean = self.running_mean
        #     std = self.running_std
        # else:
        #     # Calculate mean and variance during training
        #     x_pool_dim = x.view(-1, x.shape[-1])
        #     mean = torch.mean(x_pool_dim, dim=0)
        #     std = torch.std(x_pool_dim, dim=0)

        #     self.running_mean =  mean
        #     self.running_std =  std

        # Normalize the input
        #x_hat = (x_pool_dim - mean) / std

        self.running_mean =  x_pool_dim.mean(dim=0)
        self.running_std =  x_pool_dim.std(dim=0)
    
        x_hat = (x_pool_dim - x_pool_dim.mean(dim=0)) / x_pool_dim.std(dim=0)
        return x_hat.view(B,D,I)

    def inverse(self, x_hat):
        """
        Denormalizes the input tensor by reversing the normalization process.

        Args:
            x_hat: Normalized tensor.

        Returns:
            Unnormalized tensor.
        """
        mean = self.running_mean 
        std = self.running_std
        return x_hat * std + mean
    


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x




################################################################
#  standard 1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x




class FNO_standard_1D(nn.Module):
    def __init__(self, modes, width, input_size, output_size):
        super(FNO_standard_1D, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.input_size = input_size
        self.output_size = output_size
        self.fc0 = nn.Linear(self.input_size + 1, self.width) # input channel is 2: (a(x), x)
        
        self.normalizer = Normalizer_1D()
        self.noise_std = 0.01
        #self.t_embed = nn.Embedding(250,100)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)



        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, x):

        grid = self.get_grid(x.shape, device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x).permute(0,2,1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)


        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)





class FNO_standard_1D(nn.Module):
    def __init__(self, modes, width, input_size, output_size):
        super(FNO_standard_1D, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.input_size = input_size
        self.output_size = output_size
        self.fc0 = nn.Linear(self.input_size + 1, self.width) # input channel is 2: (a(x), x)
        
        self.normalizer = Normalizer_1D()
        self.noise_std = 0.01
        #self.t_embed = nn.Embedding(250,100)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)



        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, x):

        grid = self.get_grid(x.shape, device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x).permute(0,2,1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)


        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)








class FNO_standard_1D_KS1(nn.Module):
    def __init__(self, modes, width, input_size, output_size):
        super(FNO_standard_1D_KS1, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.input_size = input_size
        self.output_size = output_size
        self.fc0 = nn.Linear(self.input_size + 1, self.width) # input channel is 2: (a(x), x)
        
        self.normalizer = Normalizer_1D()
        self.noise_std = 0.01
        #self.t_embed = nn.Embedding(250,100)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)

        self.conv2_1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2_2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2_3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2_4 = SpectralConv1d(self.width, self.width, self.modes1)

        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)



        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        self.w2_1 = nn.Conv1d(self.width, self.width, 1)
        self.w2_2 = nn.Conv1d(self.width, self.width, 1)
        self.w2_3 = nn.Conv1d(self.width, self.width, 1)
        self.w2_4 = nn.Conv1d(self.width, self.width, 1)

        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, x):

        grid = self.get_grid(x.shape, device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x).permute(0,2,1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2_1(x)
        x2 = self.w2_1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2_2(x)
        x2 = self.w2_2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2_3(x)
        x2 = self.w2_3(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2_4(x)
        x2 = self.w2_4(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)









# class FNO_standard_1D(nn.Module):
#     def __init__(self, modes, width, input_size, output_size):
#         super(FNO_standard_1D, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .

#         input: the solution of the initial condition and location (a(x), x)
#         input shape: (batchsize, x=s, c=2)
#         output: the solution of a later timestep
#         output shape: (batchsize, x=s, c=1)
#         """

#         self.modes1 = modes
#         self.width = width
#         self.input_size = input_size
#         self.output_size = output_size
#         self.fc0 = nn.Linear(self.input_size + 1, self.width) # input channel is 2: (a(x), x)
        
#         self.normalizer = Normalizer_1D()
#         self.noise_std = 0.01
#         #self.t_embed = nn.Embedding(250,100)

#         self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.w0 = nn.Conv1d(self.width, self.width, 1)
#         self.w1 = nn.Conv1d(self.width, self.width, 1)
#         self.w2 = nn.Conv1d(self.width, self.width, 1)
#         self.w3 = nn.Conv1d(self.width, self.width, 1)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, self.output_size)

#         self.f_final_1 = nn.Linear(100,50)
#         self.f_final_2 = nn.Linear(50,1)

#     def forward(self, x):

#         grid = self.get_grid(x.shape, device)
#         x = torch.cat((x, grid), dim=-1)

#         x = self.fc0(x).permute(0,2,1)

#         x1 = self.conv0(x)
#         x2 = self.w0(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv1(x)
#         x2 = self.w1(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv2(x)
#         x2 = self.w2(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv3(x)
#         x2 = self.w3(x)
#         x = x1 + x2

#         x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)


#         x = self.f_final_2(F.relu(self.f_final_1(x.squeeze(-1))))
#         return x


#     def get_grid(self, shape, device):
#         batchsize, size_x = shape[0], shape[1]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
#         return gridx.to(device)








# class FNO_standard_1D(nn.Module):
#     def __init__(self, modes, width, input_size, output_size):
#         super(FNO_standard_1D, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .

#         input: the solution of the initial condition and location (a(x), x)
#         input shape: (batchsize, x=s, c=2)
#         output: the solution of a later timestep
#         output shape: (batchsize, x=s, c=1)
#         """

#         self.modes1 = modes
#         self.width = width
#         self.input_size = input_size
#         self.output_size = output_size
#         self.fc0 = nn.Linear(self.input_size + 1, self.width) # input channel is 2: (a(x), x)

#         self.normalizer = Normalizer_1D()
#         self.noise_std = 0.01
#         #self.t_embed = nn.Embedding(250,100)

#         self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.w0 = nn.Conv1d(self.width, self.width, 1)
#         self.w1 = nn.Conv1d(self.width, self.width, 1)
#         self.w2 = nn.Conv1d(self.width, self.width, 1)
#         self.w3 = nn.Conv1d(self.width, self.width, 1)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, self.output_size)

#     def forward(self, x):

#         x = self.normalizer(x)
#         x = x + torch.randn(x.shape, device=x.device) * self.noise_std

#         grid = self.get_grid(x.shape, device)
#         x = torch.cat((x, grid), dim=-1)

#         x = self.fc0(x).permute(0,2,1)

#         x1 = self.conv0(x)
#         x2 = self.w0(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv1(x)
#         x2 = self.w1(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv2(x)
#         x2 = self.w2(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv3(x)
#         x2 = self.w3(x)
#         x = x1 + x2

#         x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)

#         x = self.normalizer.inverse(x)

#         return x

#     def get_grid(self, shape, device):
#         batchsize, size_x = shape[0], shape[1]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
#         return gridx.to(device)














# ################################################################
# ###  DecomposedSpectralConv2d
# ################################################################
# class DecomposeSpectralConv1d(nn.Module):
#     def __init__(self, in_dim, out_dim, n_modes, resdiual=True, dropout=0.1):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.n_modes = n_modes
#         self.linear = nn.Linear(in_dim, out_dim)
#         self.residual = resdiual
#         self.act = nn.ReLU(inplace=True)

#         fourier_weight = [nn.Parameter(torch.FloatTensor(
#             in_dim, out_dim, n_modes, 2)) for _ in range(2)]
#         self.fourier_weight = nn.ParameterList(fourier_weight)
#         for param in self.fourier_weight:
#             nn.init.xavier_normal_(param, gain=1/(in_dim*out_dim))

#     @staticmethod
#     def complex_matmul_1d(a, b):
#         # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
#         op = partial(torch.einsum, "bix,iox->box")
#         return torch.stack([
#             op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
#             op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
#         ], dim=-1)

#     def forward(self, x):
#         # x.shape == [batch_size, grid_size, in_dim]
#         #import pdb; pdb.set_trace()
#         B, N, I = x.shape
#         if self.residual:
#             res = self.linear(x)
#             # res.shape == [batch_size, grid_size, grid_size, out_dim]

#         x = rearrange(x, 'b n i -> b i n')
#         # x.shape == [batch_size, in_dim, grid_size, grid_size]

#         x_ft = torch.fft.rfft(x, norm='ortho')
#         # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

#         x_ft = torch.stack([x_ft.real, x_ft.imag], dim=3)
#         # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

#         out_ft = torch.zeros(B, I, N // 2 + 1, 2, device=x.device)
#         # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

#         out_ft[:, :, :self.n_modes] = self.complex_matmul_1d(
#             x_ft[:, : , :self.n_modes], self.fourier_weight[0])

#         out_ft[:, :, :self.n_modes] = self.complex_matmul_1d(
#             x_ft[:, :, :self.n_modes], self.fourier_weight[1])

#         out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

#         x = torch.fft.irfft(out_ft, norm='ortho')
#         # x.shape == [batch_size, in_dim, grid_size, grid_size]

#         x = rearrange(x, 'b i n -> b n i')
#         # x.shape == [batch_size, grid_size, grid_size, out_dim]

#         if self.residual:
#             x = self.act(x + res)
#         else:
#             x = self.act(self.linear(x))
#         return x



# class FNO_standard_1D_decompose(nn.Module):
#     def __init__(self, modes1, width, input_dim, output_dim, n_layers, dropout=0.1, residual=False, conv_residual=True):
#         super(FNO_standard_1D_decompose, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .

#         input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
#         input shape: (batchsize, x=64, y=64, c=12)
#         output: the solution of the next timestep
#         output shape: (batchsize, x=64, y=64, c=1)
#         """

#         self.modes1 = modes1
#         self.width = width
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.in_proj = nn.Linear(input_dim, self.width)
#         self.residual = residual
#         # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

#         self.spectral_layers = nn.ModuleList([])
#         for i in range(n_layers):
#             self.spectral_layers.append(DecomposeSpectralConv1d(in_dim=width,
#                                                        out_dim=width,
#                                                        n_modes=modes1,
#                                                        resdiual=conv_residual,
#                                                        dropout=dropout))

#         self.feedforward = nn.Sequential(
#             nn.Linear(self.width, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, self.output_dim))

#     def forward(self, x, **kwargs):
#         # x.shape == [n_batches, *dim_sizes, input_size]
#         #import pdb; pdb.set_trace()
#         x = self.in_proj(x)
#         for layer in self.spectral_layers:
#             x = layer(x) + x if self.residual else layer(x)
#             #import pdb; pdb.set_trace()

#         x = self.feedforward(x)
#         # x.shape == [n_batches, *dim_sizes, 1]
#         return x









################################################################
###  DecomposedSpectralConv2d second
################################################################
class DecomposeSpectralConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes


        fourier_weight = [nn.Parameter(torch.FloatTensor(
            in_dim, out_dim, n_modes, 2)) for _ in range(2)]
        self.fourier_weight = nn.ParameterList(fourier_weight)
        for param in self.fourier_weight:
            nn.init.xavier_normal_(param, gain=1/(in_dim*out_dim))
            

    @staticmethod
    def complex_matmul_1d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bix,iox->box")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        B, I, N = x.shape
        # x.shape == [batch_size, in_dim, grid_size]

        x_ft = torch.fft.rfft(x)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=3)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft = torch.zeros(B, I, N // 2 + 1, 2, device=x.device)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :self.n_modes] = self.complex_matmul_1d(
            x_ft[:, : , :self.n_modes], self.fourier_weight[0])

        out_ft[:, :, :self.n_modes] = self.complex_matmul_1d(
            x_ft[:, :, :self.n_modes], self.fourier_weight[1])

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x



class FNO_standard_1D_decompose(nn.Module):
    def __init__(self, modes1, width, input_dim, output_dim):
        super(FNO_standard_1D_decompose, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes1
        self.width = width
        self.input_size = input_dim
        self.output_size = output_dim
        self.fc0 = nn.Linear(self.input_size + 1, self.width) # input channel is 2: (a(x), x)

        self.conv0 = DecomposeSpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = DecomposeSpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = DecomposeSpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = DecomposeSpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        grid = self.get_grid(x.shape, device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x).permute(0,2,1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)











class FNO1d_t_concat(nn.Module):
    def __init__(self, modes, width, input_size, output_size, x_res):
        super(FNO1d_t_concat, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.input_size = input_size
        self.output_size = output_size
        self.x_res = x_res

        self.fc0 = nn.Linear(self.input_size + 1, self.width) # input channel is 2: (a(x), x)

        self.time_embed_x = nn.Sequential(
            nn.Linear(1, self.x_res//2),
            nn.ReLU(),
            nn.Linear(self.x_res//2, self.x_res)
        )

        self.time_embed_y = nn.Sequential(
            nn.Linear(1, self.x_res//2),
            nn.ReLU(),
            nn.Linear(self.x_res//2, self.x_res)
        )

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, x, x_t, y_t):
        #p.print(f"x, x_t, y_t --->{x.shape, x_t.shape, y_t.shape}")
        grid = self.get_grid(x.shape, device)

        x = torch.cat((x, grid), dim=-1)

        x_t = self.time_embed_x(x_t.permute(0,2,1)[...,0:1]).permute(0,2,1)
        y_t = self.time_embed_y(y_t.permute(0,2,1)[...,0:1]).permute(0,2,1)


        #x_t = x_t.repeat(1,self.x_res,1)#permute(0,2,1)[...,0:1]).permute(0,2,1)
        #y_t =  y_t.repeat(1,self.x_res,1) #self.time_embed_y(y_t.permute(0,2,1)[...,0:1]).permute(0,2,1)

        #p.print(f"x, x_t, y_t --->{x.shape, x_t.shape, y_t.shape}")

        x = torch.cat((x,x_t,y_t), dim=-1)

        x = self.fc0(x).permute(0,2,1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)














# class MultiheadAttention(nn.Module):
#   def __init__(self, d_model, nhead, d_k=None, d_v=None):
#     super(MultiheadAttention, self).__init__()
#     if d_k is None:
#       d_k = d_model // nhead  # Dimension of key and value (adjustable)
#     if d_v is None:
#       d_v = d_k  # Can be different, but often the same
#     self.nhead = nhead
#     self.d_k = d_k
#     self.wq = nn.Linear(d_model, nhead * d_k, bias=False)
#     self.wk = nn.Linear(d_model, nhead * d_k, bias=False)
#     self.wv = nn.Linear(d_model, nhead * d_v, bias=False)
#     self.wo = nn.Linear(nhead * d_v, d_model, bias=True)
#     self.softmax = nn.Softmax(dim=-1)

#   def forward(self, q, k, v, mask=None):
#     # Project input using query, key, and value weights
#     batch_size = q.size(0)
#     q_head = self.wq(q).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
#     k_head = self.wk(k).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
#     v_head = self.wv(v).view(batch_size, -1, self.nhead, self.d_v).transpose(1, 2)

#     # Calculate attention scores
#     scores = torch.einsum('bild,bjld->bilj', q_head, k_head.transpose(-2, -1)) / self.d_k**0.5

#     # Apply mask (optional)
#     if mask is not None:
#       scores = scores.masked_fill(mask == 0, -1e9)  # Set masked elements to -inf

#     # Apply softmax to normalize attention scores
#     attention = self.softmax(scores)

#     # Context vector as weighted sum of values
#     context_head = torch.einsum('bilj,bjld->bild', attention, v_head)

#     # Concatenate heads and project back to original dimension
#     context = context_head.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_v)
#     output = self.wo(context)

#     return output, attention  # You can return only output if needed





# class SelfAttention(nn.Module):
#   def __init__(self, d_model):
#     super(SelfAttention, self).__init__()
#     self.d_k = d_model  # Dimension of key and value vectors (adjustable)
#     self.wq = nn.Linear(d_model, self.d_k, bias=False)  # Query projection
#     self.wk = nn.Linear(d_model, self.d_k, bias=False)  # Key projection
#     self.wv = nn.Linear(d_model, self.d_k, bias=False)  # Value projection
#     self.softmax = nn.Softmax(dim=-1)  # Softmax for attention scores
#     self.attention = None

#   def forward(self, x,  x_t, y_t):
#     # Project input using query, key, and value weights

#     q = self.wq( y_t )
#     k = self.wk( x + x_t )
#     #print(self.v_up(x.permute(0,2,1)).permute(0,2,1).shape)
#     v = self.wv(x)

#     # Calculate attention scores
#     scores  = torch.matmul(q, k.transpose(-2, -1)) / self.d_k**0.5  # Scaled dot-product attention

#     # Apply softmax to normalize attention scores
#     self.attention = self.softmax(scores)

#     # Context vector as weighted sum of values
#     context = torch.matmul( self.attention, v)

#     # Return context vector and attention scores (optional)
#     return context #, attention  # You can return only context if needed
  



# def positional_encoding(d_model, max_len=250):
#     """
#     This function generates a positional encoding matrix for a given maximum sequence length and embedding dimension.

#     Args:
#         d_model: The dimension of the word embeddings (e.g., 512).
#         max_len: The maximum sequence length the model can handle (e.g., 5000).

#     Returns:
#         A PyTorch tensor with shape (max_len, d_model) representing the positional encoding matrix.
#     """

#     position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#     div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

#     pe = torch.zeros(max_len, d_model)
#     pe[:, 0::2] = torch.sin(position * div_term)
#     pe[:, 1::2] = torch.cos(position * div_term)

#     return pe



# class FNO1d_t_attention(nn.Module):
#     def __init__(self, modes, width, input_size, output_size, t_res, x_res):
#         super(FNO1d_t_attention, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .

#         input: the solution of the initial condition and location (a(x), x)
#         input shape: (batchsize, x=s, c=2)
#         output: the solution of a later timestep
#         output shape: (batchsize, x=s, c=1)
#         """

#         self.modes1 = modes
#         self.width = width
#         self.input_size = input_size
#         self.output_size = output_size
#         self.t_res = t_res,
#         self.x_res = x_res,
#         #import pdb; pdb.set_trace()
#         #self.attend_00 = SelfAttention(self.x_res)

#         self.fc0 = nn.Linear(self.input_size + 1, self.width) # input channel is 2: (a(x), x)

#         #self.t_embed = nn.Embedding(self.t_res[0], self.x_res[0])
#         self.temp_encode = positional_encoding(self.t_res[0], self.x_res[0])

#         #self.t_embed_high = nn.Linear(self.input_size + 1,self.width)

#         self.t_embed_high_in = nn.Linear(self.input_size + 1,self.width)
#         self.t_embed_high_out = nn.Linear(self.input_size + 1,self.width)

#         self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.attend_0 = SelfAttention(self.x_res[0])

#         self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.attend_1 = SelfAttention(self.x_res[0])

#         self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.attend_2 = SelfAttention(self.x_res[0])

#         self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.attend_3 = SelfAttention(self.x_res[0])

#         self.w0 = nn.Conv1d(self.width, self.width, 1)
#         self.w1 = nn.Conv1d(self.width, self.width, 1)
#         self.w2 = nn.Conv1d(self.width, self.width, 1)
#         self.w3 = nn.Conv1d(self.width, self.width, 1)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, self.output_size)

#     def forward(self, x, x_t, y_t):

#         #import pdb; pdb.set_trace()
#         grid = self.get_grid(x.shape, device)


#         x_t = self.temp_encode.permute(1,0)[x_t].to(device)
#         y_t = self.temp_encode.permute(1,0)[y_t].to(device)


#         x = self.fc0( torch.cat((x, grid), dim=-1) ).permute(0,2,1)
#         x_t = self.t_embed_high_in( torch.cat((x_t.permute(0,2,1), grid), dim=-1)).permute(0,2,1)
#         y_t = self.t_embed_high_out( torch.cat((y_t.permute(0,2,1), grid), dim=-1)).permute(0,2,1)


#         x1 = self.conv0(x)
#         #x1 = self.conv0(self.attend_0(x,x_t,y_t))
#         #x1 = self.attend_0(self.conv0(x),x_t,y_t)
#         x2 = self.w0(x)
#         x3 = self.attend_0(x,x_t,y_t)
        
#         x = x1 + x2 + x3
#         x = F.gelu(x)

#         x1 = self.conv1(x)
#         #x1 = self.conv1(self.attend_1(x,x_t,y_t))
#         #x1 = self.attend_1(self.conv1(x),x_t,y_t)
#         x2 = self.w1(x)
#         x3 = self.attend_1(x,x_t,y_t)
#         x = x1 + x2 + x3
#         x = F.gelu(x)

#         x1 = self.conv2(x)
#         #x1 = self.conv2(self.attend_2(x,x_t,y_t))
#         #x1 = self.attend_2(self.conv2(x),x_t,y_t)
#         x2 = self.w2(x)
#         x3 = self.attend_2(x,x_t,y_t)
#         x = x1 + x2 + x3
#         x = F.gelu(x)

#         x1 = self.conv3(x)
#         #x1 = self.conv3(self.attend_3(x,x_t,y_t))
#         #x1 = self.attend_3(self.conv3(x),x_t,y_t)
#         x2 = self.w3(x)
#         x3 = self.attend_3(x,x_t,y_t)
#         x = x1 + x2 + x3

#         x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         return x

#     def get_grid(self, shape, device):
#         batchsize, size_x = shape[0], shape[1]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
#         return gridx.to(device)




###########################################################################################
######### ATTENTION....
########
############################

# class SelfAttention(nn.Module):
#   def __init__(self, d_model, d_k, nhead, mask):
#     super(SelfAttention, self).__init__()

#     self.nhead = nhead
#     self.d_model = d_model
#     self.d_k = d_k  # Dimension of key and value vectors (adjustable)

#     self.wq = nn.Linear(self.d_model, self.d_k, bias=False)  # Query projection
#     self.wk = nn.Linear(self.d_model, self.d_k, bias=False)  # Key projection
#     self.wv = nn.Linear(self.d_model, self.d_k, bias=False)  # Value projection
#     self.softmax = nn.Softmax(dim=-1)  # Softmax for attention scores
#     self.attention = None
#     self.mask = mask

#   def forward(self, v,  q, k):
#     # Project input using query, key, and value weights
#     # print("\n")
#     # print("Self Attention...")
#     # print("x -->", v.shape)
#     # print("q -->", q.shape)
#     # print("k -->", k.shape)


#     v = v.permute(0,2,1,3)
#     q = q.permute(0,2,1,3)
#     k = k.permute(0,2,1,3)

#     # print("v -->", v.shape)
#     # print("q -->", q.shape)
#     # print("k -->", k.shape)

#     # print("x -->", x.shape)
#     # print("x_t -->", x_t.shape)
#     # print("y_t -->", y_t.shape)

#     q = self.wq( q )
#     k = self.wk( k )
#     #print(self.v_up(x.permute(0,2,1)).permute(0,2,1).shape)
#     v = self.wv(v)

#     batch_size = q.shape[0]
#     res = q.shape[1]
#     # t = q.shape[2]
#     # d = q.shape[3]

#     q_h = q.view(batch_size, res, -1, self.nhead, self.d_k//self.nhead)
#     k_h = k.view(batch_size, res, -1, self.nhead, self.d_k//self.nhead)
#     v_h = v.view(batch_size, res, -1, self.nhead, self.d_k//self.nhead)


#     # print("x -->", v_h.shape)
#     # print("q -->", q_h.shape)
#     # print("k -->", k_h.shape)



#     q_h = q_h.permute(0,3,1,2,4)
#     k_h = k_h.permute(0,3,1,2,4)
#     v_h = v_h.permute(0,3,1,2,4)

#     # print("x -->", v_h.shape)
#     # print("q -->", q_h.shape)
#     # print("k -->", k_h.shape)
#     # print("x -->", x.shape)
#     # print("x_t -->", x_t.shape)
#     # print("y_t -->", y_t.shape)
#     # Calculate attention scores
#     scores  = torch.matmul(q_h, k_h.transpose(-2, -1)) / self.d_k**0.5  # Scaled dot-product attention
#     scores = scores.masked_fill(self.mask==0, -1e9).to(device)
#     #print("scores -->", scores.shape)
#     # Apply softmax to normalize attention scores
#     self.attention = self.softmax(scores)

#     # Context vector as weighted sum of values
#     context_h = torch.matmul( self.attention, v_h)

#     #print("context -->", context.shape)
#     context_h = context_h.permute(0,2,3,1,4)
#     context = context_h.reshape(batch_size, res, -1, self.d_k)

#     q = q.permute(0,2,1,3)
#     k = k.permute(0,2,1,3)
#     context = context.permute(0,2,1,3)

#     # print("q -->", q.shape)
#     # print("k -->", k.shape)
#     # print("context -->", context.shape)
#     #print("\n")
#     # Return context vector and attention scores (optional)
#     return context, q, k #, attention  # You can return only context if needed




class SelfAttention(nn.Module):
  def __init__(self, d_model, d_k, nhead, mask):
    super(SelfAttention, self).__init__()

    self.nhead = nhead
    self.d_model = d_model
    self.d_k = d_k  # Dimension of key and value vectors (adjustable)

    self.wq = nn.Linear(self.d_model, self.d_model, bias=False)  # Query projection
    self.wk = nn.Linear(self.d_model, self.d_model, bias=False)  # Key projection
    self.wv = nn.Linear(self.d_model, self.d_model, bias=False)  # Value projection
    self.softmax = nn.Softmax(dim=-1)  # Softmax for attention scores
    self.attention = None
    self.mask = mask

  def forward(self, v, q, k):
    # Project input using query, key, and value weights
    # print("\n")
    # p.print(f"Self Attention...")
    # p.print(f"v --> {v.shape}")
    # p.print(f"q --> {q.shape}" )
    # p.print(f"k -->  {k.shape}")


    v = v.permute(0,2,1)
    q = q.permute(0,2,1)
    k = k.permute(0,2,1)

    # p.print(f"v --> {v.shape}")
    # p.print(f"q --> {q.shape}" )
    # p.print(f"k -->  {k.shape}")

    # print("x -->", x.shape)
    # print("x_t -->", x_t.shape)
    # print("y_t -->", y_t.shape)

    q = self.wq( q )
    k = self.wk( k )
    #print(self.v_up(x.permute(0,2,1)).permute(0,2,1).shape)
    v = self.wv(v)

    batch_size = q.shape[0]
    res = q.shape[1]
    # t = q.shape[2]
    # d = q.shape[3]

    q_h = q.view(batch_size, -1, self.nhead, self.d_k//self.nhead)
    k_h = k.view(batch_size, -1, self.nhead, self.d_k//self.nhead)
    v_h = v.view(batch_size, -1, self.nhead, self.d_k//self.nhead)


    # p.print(f"v --> {v_h.shape}")
    # p.print(f"q --> {q_h.shape}" )
    # p.print(f"k -->  {k_h.shape}")



    q_h = q_h.permute(0,2,3,1)
    k_h = k_h.permute(0,2,3,1)
    v_h = v_h.permute(0,2,3,1)


    # q_h = q_h.permute(0,2,1,3)
    # k_h = k_h.permute(0,2,1,3)
    # v_h = v_h.permute(0,2,1,3)

    # p.print(f"v --> {v_h.shape}")
    # p.print(f"q --> {q_h.shape}" )
    # p.print(f"k -->  {k_h.shape}")

    # print("x -->", x.shape)
    # print("x_t -->", x_t.shape)
    # print("y_t -->", y_t.shape)
    # Calculate attention scores
    scores  = torch.matmul(q_h, k_h.transpose(-2, -1)) / self.d_k**0.5  # Scaled dot-product attention
    scores = scores.masked_fill(self.mask==0, -1e9).to(device)
    #p.print(f"scores --> {scores.shape}")
    # Apply softmax to normalize attention scores
    self.attention = self.softmax(scores)

    # Context vector as weighted sum of values
    context_h = torch.matmul( self.attention, v_h)
    #p.print(f"context_h --> {context_h.shape}")

    context_h = context_h.permute(0,3,1,2)
    #context_h = context_h.permute(0,2,1,3)

    context = context_h.reshape(batch_size, -1, self.d_k)

    q = q.permute(0,2,1)
    k = k.permute(0,2,1)
    context = context.permute(0,2,1)

    #p.print(f"context --> {context.shape}")
    #p.print(f"q --> {q.shape}" )
    #p.print(f"k -->  {k.shape}")


    #print("\n")
    # Return context vector and attention scores (optional)
    return context, q, k #, attention  # You can return only context if needed



############################################################################
########################################################################

# class SpectralConv1d_t(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1):
#         super(SpectralConv1d_t, self).__init__()

#         """
#         1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
#         """

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.t_step = 10
#         self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

#         self.scale = (1 / (in_channels*out_channels))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(self.t_step, in_channels, out_channels, self.modes1, dtype=torch.cfloat))

#     # Complex multiplication
#     def compl_mul1d(self, input, weights):
#         # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
#         return torch.einsum("btix,tiox->btox", input, weights)

#     def forward(self, x):
#         #print("Spec Conv ....")
#         #print( "x -->", x.shape)
#         x = x.permute(0,1,3,2)
#         batchsize = x.shape[0]
#         #Compute Fourier coeffcients up to factor of e^(- something constant)
#         x_ft = torch.fft.rfft(x)

#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.t_step, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
#         #print("out_ft -->", out_ft.shape)
#         #print("self.weights1 -->", self.weights1.shape)
#         #print("x_ft -->", x_ft[:, :, :, :self.modes1].shape)
#         out_ft[:, :, :, :self.modes1] = self.compl_mul1d(x_ft[:, :,:, :self.modes1], self.weights1)

#         #Return to physical space
#         x = torch.fft.irfft(out_ft, n=x.size(-1))
#         x = x.permute(0,1,3,2)
#         #print("x ->", x.shape)
#         return x
    


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x




############################################################################
########################################################################

# class FNO1d_t_attention(nn.Module):
#     def __init__(self, modes, width, input_size, output_size, t_res, x_res, nhead):
#         super(FNO1d_t_attention, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .

#         input: the solution of the initial condition and location (a(x), x)
#         input shape: (batchsize, x=s, c=2)
#         output: the solution of a later timestep
#         output shape: (batchsize, x=s, c=1)
#         """

#         self.modes1 = modes
#         self.width = width
#         self.input_size = input_size
#         self.output_size = output_size
#         self.t_res = t_res,
#         self.x_res = x_res,
#         self.nhead = nhead
#         #import pdb; pdb.set_trace()
#         #self.attend_00 = SelfAttention(self.x_res)

#         self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

#         #self.t_embed = nn.Embedding(self.t_res[0], self.x_res[0])
#         #self.temp_encode = positional_encoding(self.t_res[0], self.x_res[0])
#         #self.temp_encode = nn.Linear(1, self.x_res[0])

#         #self.t_embed_high = nn.Linear(self.input_size + 1,self.width)

#         self.t_embed_high_in = nn.Linear(2,self.width)
#         self.t_embed_high_out = nn.Linear(2,self.width)

#         self.conv0 = SpectralConv1d_t(self.width, self.width, self.modes1)
#         self.attend_0 = SelfAttention(self.width, self.width, self.nhead )

#         self.conv1 = SpectralConv1d_t(self.width, self.width, self.modes1)
#         self.attend_1 = SelfAttention(self.width, self.width, self.nhead )

#         self.conv2 = SpectralConv1d_t(self.width, self.width, self.modes1)
#         self.attend_2 = SelfAttention(self.width, self.width, self.nhead )

#         self.conv3 = SpectralConv1d_t(self.width, self.width, self.modes1)
#         self.attend_3 = SelfAttention(self.width, self.width, self.nhead )

#         self.w0 = nn.Conv1d(self.width, self.width, 1)
#         self.w1 = nn.Conv1d(self.width, self.width, 1)
#         self.w2 = nn.Conv1d(self.width, self.width, 1)
#         self.w3 = nn.Conv1d(self.width, self.width, 1)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x, x_t, y_t):

#         #import pdb; pdb.set_trace()
#         #print("x ->", x.shape)
#         x = x.permute(0,2,1)
#         x_t = x_t.permute(0,2,1)
#         y_t = y_t.permute(0,2,1)
#         # x_t = x_t[:, 0, :]
#         # y_t = y_t[:, 0, :]


#         x = x.unsqueeze(-1)
#         grid = self.get_grid(x.shape, device)
#         # print("x -->", x.shape)
#         # print("grid -->",grid.shape)

#         # print("x -->", x.shape)
#         # print("x_t -->", x_t.shape)
#         # print("y_t -->", y_t.shape)
#         # x_t = self.temp_encode.permute(1,0)[x_t].to(device)
#         # y_t = self.temp_encode.permute(1,0)[y_t].to(device)

#         #x_t = self.temp_encode( x_t.unsqueeze(-1) ).to(device)
#         x_t = x_t.unsqueeze(-1)
#         #y_t = self.temp_encode( y_t.unsqueeze(-1) ).to(device)
#         y_t = y_t.unsqueeze(-1)


#         # print("x -->", x.shape)
#         # print("x_t -->", x_t.shape)
#         # print("y_t -->", y_t.shape)

#         x = self.fc0( torch.cat((x, grid), dim=-1) )#.permute(0,2,1)
#         x_t = self.t_embed_high_in( torch.cat((x_t, grid), dim=-1))#.permute(0,2,1)
#         y_t = self.t_embed_high_out( torch.cat((y_t, grid), dim=-1))#.permute(0,2,1)


#         # print("x -->", x.shape)
#         # print("x_t -->", x_t.shape)
#         # print("y_t -->", y_t.shape)

#         #x1 = self.conv0(x)
#         #x1 = self.conv0(self.attend_0(x,x_t,y_t))
#         #x1 = self.attend_0(self.conv0(x),x_t,y_t)
#         x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         x2 = self.w0(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         x3, q, k = self.attend_0(x, x + y_t, x + x_t)
#         x1 = self.conv0(x3)
#         x = x1 + x2 + x3
#         x = F.gelu(x)


#         #x1 = self.conv1(x)
#         #x1 = self.conv1(self.attend_1(x,x_t,y_t))
#         #x1 = self.attend_1(self.conv1(x),x_t,y_t)
#         #x2 = self.w1(x)
#         x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         x2 = self.w1(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         x3, q, k = self.attend_1(x, q, k)
#         x1 = self.conv1(x3)
#         x = x1 + x2 + x3
#         x = F.gelu(x)


#         #x1 = self.conv2(x)
#         #x1 = self.conv2(self.attend_2(x,x_t,y_t))
#         #x1 = self.attend_2(self.conv2(x),x_t,y_t)
#         #x2 = self.w2(x)
#         x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         x2 = self.w2(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         x3, q, k = self.attend_2(x, q, k)
#         x1 = self.conv2(x3)
#         x = x1 + x2 + x3
#         x = F.gelu(x)


#         #x1 = self.conv3(x)
#         #x1 = self.conv3(self.attend_3(x,x_t,y_t))
#         #x1 = self.attend_3(self.conv3(x),x_t,y_t)
#         #x2 = self.w3(x)
#         x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         x2 = self.w3(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         x3, q, k = self.attend_3(x, q, k)
#         x1 = self.conv3(x3)
#         x = x1 + x2 + x3


#         #x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         #print(x.shape)


#         x = x.squeeze(-1).permute(0,2,1)
#         return x

#     def get_grid(self, shape, device):
#         batchsize, size_t, size_x = shape[0], shape[1], shape[2]
#         #gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = torch.linspace(0, 1, size_x)
#         gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_t, 1, 1])
#         return gridx.to(device)








# class FNO1d_t_attention(nn.Module):
#     def __init__(self, modes, width, input_size, output_size, t_res, x_res, nhead):
#         super(FNO1d_t_attention, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .

#         input: the solution of the initial condition and location (a(x), x)
#         input shape: (batchsize, x=s, c=2)
#         output: the solution of a later timestep
#         output shape: (batchsize, x=s, c=1)
#         """

#         self.modes1 = modes
#         self.width = width
#         self.input_size = input_size
#         self.output_size = output_size
#         self.t_res = t_res
#         self.x_res = x_res
#         self.nhead = nhead
#         #import pdb; pdb.set_trace()
#         #self.attend_00 = SelfAttention(self.x_res)

#         self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

#         #self.t_embed = nn.Embedding(self.t_res[0], self.x_res[0])
#         #self.temp_encode = positional_encoding(self.t_res[0], self.x_res[0])
#         #self.temp_encode = nn.Linear(1, self.x_res[0])

#         #self.t_embed_high = nn.Linear(self.input_size + 1,self.width)

#         self.t_embed_high = nn.Linear(2,self.width)
#         #self.t_embed_high_in = nn.Linear(2,self.width)
#         #self.t_embed_high_out = nn.Linear(2,self.width)

#         self.conv0 = SpectralConv1d_t(self.width, self.width, self.modes1)
#         self.attend_0 = SelfAttention(self.width, self.width, self.nhead )
#         self.layernorm_0 = nn.LayerNorm([self.width, self.input_size, self.x_res ], elementwise_affine=False)
#         #self.ffn_0 = nn.Sequential(nn.Linear(self.width, self.width), nn.GELU(), nn.Linear(self.width, self.width))

#         self.conv1 = SpectralConv1d_t(self.width, self.width, self.modes1)
#         self.attend_1 = SelfAttention(self.width, self.width, self.nhead )
#         self.layernorm_1 = nn.LayerNorm([self.width, self.input_size, self.x_res ], elementwise_affine=False)
#         #self.ffn_1 = nn.Sequential(nn.Linear(self.width, self.width), nn.GELU(), nn.Linear(self.width, self.width))

#         self.conv2 = SpectralConv1d_t(self.width, self.width, self.modes1)
#         self.attend_2 = SelfAttention(self.width, self.width, self.nhead )
#         self.layernorm_2 = nn.LayerNorm([self.width, self.input_size, self.x_res ], elementwise_affine=False)
#         #self.ffn_2 = nn.Sequential(nn.Linear(self.width, self.width), nn.GELU(), nn.Linear(self.width, self.width))

#         self.conv3 = SpectralConv1d_t(self.width, self.width, self.modes1)
#         self.attend_3 = SelfAttention(self.width, self.width, self.nhead )
#         self.layernorm_3 = nn.LayerNorm([self.width, self.input_size, self.x_res ], elementwise_affine=False)
#         #self.ffn_3 = nn.Sequential(nn.Linear(self.width, self.width), nn.GELU(), nn.Linear(self.width, self.width))

#         self.w0 = nn.Conv1d(self.width, self.width, 1)
#         self.w1 = nn.Conv1d(self.width, self.width, 1)
#         self.w2 = nn.Conv1d(self.width, self.width, 1)
#         self.w3 = nn.Conv1d(self.width, self.width, 1)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)

#         p.print(f"[self.width, self.input_size, self.x_res ] :  {[self.width, self.input_size, self.x_res ]}")

#     def forward(self, x, x_t, y_t):

#         #import pdb; pdb.set_trace()
#         #print("x ->", x.shape)
#         x = x.permute(0,2,1)
#         x_t = x_t.permute(0,2,1)
#         y_t = y_t.permute(0,2,1)
#         # x_t = x_t[:, 0, :]
#         # y_t = y_t[:, 0, :]


#         x = x.unsqueeze(-1)
#         grid = self.get_grid(x.shape, device)
#         # print("x -->", x.shape)
#         # print("grid -->",grid.shape)

#         # print("x -->", x.shape)
#         # print("x_t -->", x_t.shape)
#         # print("y_t -->", y_t.shape)
#         # x_t = self.temp_encode.permute(1,0)[x_t].to(device)
#         # y_t = self.temp_encode.permute(1,0)[y_t].to(device)

#         #x_t = self.temp_encode( x_t.unsqueeze(-1) ).to(device)
#         x_t = x_t.unsqueeze(-1)
#         #y_t = self.temp_encode( y_t.unsqueeze(-1) ).to(device)
#         y_t = y_t.unsqueeze(-1)


#         # print("x -->", x.shape)
#         # print("x_t -->", x_t.shape)
#         # print("y_t -->", y_t.shape)

#         x = self.fc0( torch.cat((x, grid), dim=-1) )#.permute(0,2,1)
#         x_t = self.t_embed_high( torch.cat((x_t, grid), dim=-1))#.permute(0,2,1)
#         y_t = self.t_embed_high( torch.cat((y_t, grid), dim=-1))#.permute(0,2,1)


#         # print("x -->", x.shape)
#         # print("x_t -->", x_t.shape)
#         # print("y_t -->", y_t.shape)

#         #x1 = self.conv0(x)
#         #x1 = self.conv0(self.attend_0(x,x_t,y_t))
#         #x1 = self.attend_0(self.conv0(x),x_t,y_t)
#         # x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         # x2 = self.w0(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         # x3, q, k = self.attend_0(x, x + y_t, x + x_t)
#         # x1 = self.conv0(x3)
#         # x = x1 + x2 + x3
#         # x = F.gelu(x)





#         x = x + self.layernorm_0(x.permute(0,3,1,2)).permute(0,2,3,1)
#         x3, q, k = self.attend_0(x, x + y_t, x + x_t)

#         # x3 = x3 + self.layernorm_0(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         # x3 = self.ffn_0(x3)

#         #x3 = x3 + self.layernorm_0(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         #x1 = self.conv0(x3)

#         x1 = self.conv0(x)

#         x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         x2 = self.w0(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         #x = x1 + x2
#         x = x1 + x2 + x3
#         x = F.gelu(x)




#         x = x + self.layernorm_1(x.permute(0,3,1,2)).permute(0,2,3,1)
#         x3, q, k = self.attend_1(x, x + y_t, x + x_t)

#         # x3 = x3 + self.layernorm_1(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         # x3 = self.ffn_1(x3)

#         # x3 = x3 + self.layernorm_1(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         # x1 = self.conv1(x3)
#         x1 = self.conv1(x)

#         x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         x2 = self.w1(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         #x = x1 + x2
#         x = x1 + x2 + x3
#         x = F.gelu(x)



#         x = x + self.layernorm_2(x.permute(0,3,1,2)).permute(0,2,3,1)
#         x3, q, k = self.attend_2(x, x + y_t, x + x_t)

#         # x3 = x3 + self.layernorm_2(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         # x3 = self.ffn_2(x3)

#         # x3 = x3 + self.layernorm_2(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         # x1 = self.conv2(x3)

#         x1 = self.conv2(x)

#         x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         x2 = self.w2(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         #x = x1 + x2
#         x = x1 + x2 + x3
#         x = F.gelu(x)



#         x = x + self.layernorm_3(x.permute(0,3,1,2)).permute(0,2,3,1)
#         x3, q, k = self.attend_3(x, x + y_t, x + x_t)

#         # x3 = x3 + self.layernorm_3(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         # x3 = self.ffn_3(x3)

#         # x3 = x3 + self.layernorm_3(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         # x1 = self.conv2(x3)

#         x1 = self.conv3(x)


#         x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         x2 = self.w3(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         #x = x1 + x2
#         x = x1 + x2 + x3


#         #x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         #print(x.shape)
#         x = x.squeeze(-1).permute(0,2,1)
#         return x

#     def get_grid(self, shape, device):
#         batchsize, size_t, size_x = shape[0], shape[1], shape[2]
#         #gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = torch.linspace(0, 1, size_x)
#         gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_t, 1, 1])
#         return gridx.to(device)




















# class FNO1d_t_attention(nn.Module):
#     def __init__(self, modes, width, input_size, output_size, t_res, x_res, nhead):
#         super(FNO1d_t_attention, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .

#         input: the solution of the initial condition and location (a(x), x)
#         input shape: (batchsize, x=s, c=2)
#         output: the solution of a later timestep
#         output shape: (batchsize, x=s, c=1)
#         """

#         self.modes1 = modes
#         self.width = width
#         self.input_size = input_size
#         self.output_size = output_size
#         self.t_res = t_res
#         self.x_res = x_res
#         self.nhead = nhead
#         #import pdb; pdb.set_trace()
#         #self.attend_00 = SelfAttention(self.x_res)

#         self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)
#         self.mask = torch.triu(torch.ones(10,10), diagonal=1).type(torch.uint8).to(device) == 0

#         #self.t_embed = nn.Embedding(self.t_res[0], self.x_res[0])
#         #self.temp_encode = positional_encoding(self.t_res[0], self.x_res[0])
#         #self.temp_encode = nn.Linear(1, self.x_res[0])

#         #self.t_embed_high = nn.Linear(self.input_size + 1,self.width)

#         self.t_embed_high = nn.Linear(2,self.width)
#         #self.t_embed_high_in = nn.Linear(2,self.width)
#         #self.t_embed_high_out = nn.Linear(2,self.width)

#         self.conv0 = SpectralConv1d_t(self.width, self.width, self.modes1)
#         self.attend_0 = SelfAttention(self.width, self.width, self.nhead, self.mask)
#         self.layernorm_0 = nn.LayerNorm([self.width, self.input_size, self.x_res ], elementwise_affine=False)
#         #self.ffn_0 = nn.Sequential(nn.Linear(self.width, self.width), nn.GELU(), nn.Linear(self.width, self.width))

#         self.conv1 = SpectralConv1d_t(self.width, self.width, self.modes1)
#         self.attend_1 = SelfAttention(self.width, self.width, self.nhead, self.mask )
#         self.layernorm_1 = nn.LayerNorm([self.width, self.input_size, self.x_res ], elementwise_affine=False)
#         #self.ffn_1 = nn.Sequential(nn.Linear(self.width, self.width), nn.GELU(), nn.Linear(self.width, self.width))

#         self.conv2 = SpectralConv1d_t(self.width, self.width, self.modes1)
#         self.attend_2 = SelfAttention(self.width, self.width, self.nhead, self.mask )
#         self.layernorm_2 = nn.LayerNorm([self.width, self.input_size, self.x_res ], elementwise_affine=False)
#         #self.ffn_2 = nn.Sequential(nn.Linear(self.width, self.width), nn.GELU(), nn.Linear(self.width, self.width))

#         self.conv3 = SpectralConv1d_t(self.width, self.width, self.modes1)
#         self.attend_3 = SelfAttention(self.width, self.width, self.nhead, self.mask )
#         self.layernorm_3 = nn.LayerNorm([self.width, self.input_size, self.x_res ], elementwise_affine=False)
#         #self.ffn_3 = nn.Sequential(nn.Linear(self.width, self.width), nn.GELU(), nn.Linear(self.width, self.width))

#         self.w0 = nn.Conv1d(self.width, self.width, 1)
#         self.w1 = nn.Conv1d(self.width, self.width, 1)
#         self.w2 = nn.Conv1d(self.width, self.width, 1)
#         self.w3 = nn.Conv1d(self.width, self.width, 1)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)

#         p.print(f"[self.width, self.input_size, self.x_res ] :  {[self.width, self.input_size, self.x_res ]}")

#     def forward(self, x, x_t, y_t):

#         #import pdb; pdb.set_trace()
#         #print("x ->", x.shape)
#         x = x.permute(0,2,1)
#         x_t = x_t.permute(0,2,1)
#         y_t = y_t.permute(0,2,1)
#         # x_t = x_t[:, 0, :]
#         # y_t = y_t[:, 0, :]


#         x = x.unsqueeze(-1)
#         grid = self.get_grid(x.shape, device)
#         # print("x -->", x.shape)
#         # print("grid -->",grid.shape)

#         # print("x -->", x.shape)
#         # print("x_t -->", x_t.shape)
#         # print("y_t -->", y_t.shape)
#         # x_t = self.temp_encode.permute(1,0)[x_t].to(device)
#         # y_t = self.temp_encode.permute(1,0)[y_t].to(device)

#         #x_t = self.temp_encode( x_t.unsqueeze(-1) ).to(device)
#         x_t = x_t.unsqueeze(-1)
#         #y_t = self.temp_encode( y_t.unsqueeze(-1) ).to(device)
#         y_t = y_t.unsqueeze(-1)


#         # print("x -->", x.shape)
#         # print("x_t -->", x_t.shape)
#         # print("y_t -->", y_t.shape)

#         x = self.fc0( torch.cat((x, grid), dim=-1) )#.permute(0,2,1)
#         x_t = self.t_embed_high( torch.cat((x_t, grid), dim=-1))#.permute(0,2,1)
#         y_t = self.t_embed_high( torch.cat((y_t, grid), dim=-1))#.permute(0,2,1)


#         # print("x -->", x.shape)
#         # print("x_t -->", x_t.shape)
#         # print("y_t -->", y_t.shape)

#         #x1 = self.conv0(x)
#         #x1 = self.conv0(self.attend_0(x,x_t,y_t))
#         #x1 = self.attend_0(self.conv0(x),x_t,y_t)
#         # x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         # x2 = self.w0(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         # x3, q, k = self.attend_0(x, x + y_t, x + x_t)
#         # x1 = self.conv0(x3)
#         # x = x1 + x2 + x3
#         # x = F.gelu(x)





#         x = x + self.layernorm_0(x.permute(0,3,1,2)).permute(0,2,3,1)
#         #x = self.layernorm_0(x.permute(0,3,1,2)).permute(0,2,3,1)
#         x3, q, k = self.attend_0(x, x + y_t, x + x_t)

#         # x3 = x3 + self.layernorm_0(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         # x3 = self.ffn_0(x3)

#         x3 = x3 + self.layernorm_0(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         #x3 = self.layernorm_0(x3.permute(0,3,1,2)).permute(0,2,3,1)

#         x1 = self.conv0(x3)

#         #x1 = self.conv0(x)

#         x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         x2 = self.w0(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         x = x1 + x2
#         #x = x1 + x2 + x3
#         x = F.gelu(x)



        
#         x = x + self.layernorm_1(x.permute(0,3,1,2)).permute(0,2,3,1)
#         #x = self.layernorm_1(x.permute(0,3,1,2)).permute(0,2,3,1)

#         x3, q, k = self.attend_1(x, x + y_t, x + x_t)

#         # x3 = x3 + self.layernorm_1(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         # x3 = self.ffn_1(x3)

#         x3 = x3 + self.layernorm_1(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         #x3 = self.layernorm_1(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         x1 = self.conv1(x3)

#         #x1 = self.conv1(x)

#         x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         x2 = self.w1(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         x = x1 + x2
#         #x = x1 + x2 + x3
#         x = F.gelu(x)



#         x = x + self.layernorm_2(x.permute(0,3,1,2)).permute(0,2,3,1)
#         #x = self.layernorm_2(x.permute(0,3,1,2)).permute(0,2,3,1)
#         x3, q, k = self.attend_2(x, x + y_t, x + x_t)

#         # x3 = x3 + self.layernorm_2(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         # x3 = self.ffn_2(x3)

#         x3 = x3 + self.layernorm_2(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         #x3 = self.layernorm_2(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         x1 = self.conv2(x3)

#         #x1 = self.conv2(x)

#         x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         x2 = self.w2(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         x = x1 + x2
#         #x = x1 + x2 + x3
#         x = F.gelu(x)



#         x = x + self.layernorm_3(x.permute(0,3,1,2)).permute(0,2,3,1)
#         #x = self.layernorm_3(x.permute(0,3,1,2)).permute(0,2,3,1)
#         x3, q, k = self.attend_3(x, x + y_t, x + x_t)

#         # x3 = x3 + self.layernorm_3(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         # x3 = self.ffn_3(x3)

#         x3 = x3 + self.layernorm_3(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         #x3 = self.layernorm_3(x3.permute(0,3,1,2)).permute(0,2,3,1)
#         x1 = self.conv2(x3)

#         #x1 = self.conv3(x)


#         x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
#         x2 = self.w3(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#         x = x1 + x2
#         #x = x1 + x2 + x3


#         #x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         #print(x.shape)
#         x = x.squeeze(-1).permute(0,2,1)
#         return x

#     def get_grid(self, shape, device):
#         batchsize, size_t, size_x = shape[0], shape[1], shape[2]
#         #gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = torch.linspace(0, 1, size_x)
#         gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_t, 1, 1])
#         return gridx.to(device)






class FNO1d_t_attention(nn.Module):
    def __init__(self, modes, width, input_size, output_size, t_res, x_res, nhead):
        super(FNO1d_t_attention, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.input_size = input_size
        self.output_size = output_size
        self.t_res = t_res
        self.x_res = x_res
        self.nhead = nhead

        #import pdb; pdb.set_trace()
        #self.attend_00 = SelfAttention(self.x_res)

        #self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)
        self.fc0 = nn.Linear(self.input_size+1, self.width)
        self.mask = torch.triu(torch.ones(self.width//self.nhead, self.width//self.nhead), diagonal=1).type(torch.uint8).to(device) == 0
        #self.mask = torch.triu(torch.ones(self.x_res, self.x_res), diagonal=1).type(torch.uint8).to(device) == 0

        #self.t_embed = nn.Embedding(self.t_res[0], self.x_res[0])
        #self.temp_encode = positional_encoding(self.t_res[0], self.x_res[0])

        #self.temp_encode = nn.Linear(1, self.x_res[0])

        #self.t_embed_high = nn.Linear(self.input_size + 1,self.width)

        self.t_embed_high_in = nn.Linear(self.input_size+1,self.width)
        self.t_embed_high_out = nn.Linear(self.input_size+1,self.width)

        
        self.time_embed_x = nn.Sequential(
            nn.Linear(1, self.x_res//2),
            nn.ReLU(),
            nn.Linear(self.x_res//2, self.x_res)
        )

        self.time_embed_y = nn.Sequential(
            nn.Linear(1, self.x_res//2),
            nn.ReLU(),
            nn.Linear(self.x_res//2, self.x_res)
        )



        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.attend_0 = SelfAttention(self.width, self.width, self.nhead, self.mask)
        #self.attend_0 = SelfAttention(self.width)

        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.attend_1 = SelfAttention(self.width, self.width, self.nhead, self.mask)
        #self.attend_1 = SelfAttention(self.width)

        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.attend_2 = SelfAttention(self.width, self.width, self.nhead, self.mask)
        #self.attend_2 = SelfAttention(self.width)

        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.attend_3 = SelfAttention(self.width, self.width, self.nhead, self.mask)
        #self.attend_3 = SelfAttention(self.width)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, x, x_t, y_t):

        #import pdb; pdb.set_trace()
        #print("x ->", x.shape)
        #x = x.unsqueeze(-1)
        # p.print(f"x --> {x.shape}")
        # p.print(f"x_t --> {x_t.shape}")
        # p.print(f"y_t --> {y_t.shape}")

        x_t = x_t[:, :1, :]
        y_t = y_t[:, :1, :]

        grid = self.get_grid(x[...,0:1].shape, device)
        # print("x -->", x.shape)
        #p.print(f"grid --> {grid.shape}")

        # print("x -->", x.shape)
        # print("x_t -->", x_t.shape)
        # print("y_t -->", y_t.shape)
        
        # x_t = self.temp_encode.permute(1,0)[x_t].to(device)
        # y_t = self.temp_encode.permute(1,0)[y_t].to(device)

        # x_t = self.temp_encode( x_t.unsqueeze(-1) ).to(device)
        # x_t = x_t.unsqueeze(-1)
        # y_t = self.temp_encode( y_t.unsqueeze(-1) ).to(device)
        # y_t = y_t.unsqueeze(-1)

        x_t = self.time_embed_x(x_t.permute(0,2,1)).permute(0,2,1).to(device) # potentially add x
        y_t = self.time_embed_y(y_t.permute(0,2,1)).permute(0,2,1).to(device)


        # p.print(f"x_t --> {x_t.shape}")
        # p.print(f"y_t --> {y_t.shape}")

        # print("x -->", x.shape)
        # print("x_t -->", x_t.shape)
        # print("y_t -->", y_t.shape)
        # print("grid -->", grid.shape)
        
        x_t = self.t_embed_high_in( torch.cat((x_t, grid), dim=-1)).permute(0,2,1)  #ADDEDDD..... 5
        y_t = self.t_embed_high_out( torch.cat((y_t, grid), dim=-1)).permute(0,2,1)
        x = self.fc0( torch.cat((x, grid), dim=-1) ).permute(0,2,1)

        # p.print(f"x --> {x.shape}")
        # p.print(f"x_t --> {x_t.shape}")
        # p.print(f"y_t --> {y_t.shape}")

        # print("x -->", x.shape)
        # print("x_t -->", x_t.shape)
        # print("y_t -->", y_t.shape)

        x1 = self.conv0(x)
        #print("x1 -->", x1.shape)
        #x1 = self.conv0(self.attend_0(x,x_t,y_t))
        #x1 = self.attend_0(self.conv0(x),x_t,y_t)
        # x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
        # x2 = self.w0(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        x2 = self.w0(x)

        #print("x2 -->", x2.shape)
        x3, q, k = self.attend_0(x, x + y_t, x + x_t)
        #x3, q, k = self.attend_0(x, y_t, x_t)
        x = x1 + x2  + x3
        x = F.gelu(x)

        x1 = self.conv1(x)
        #x1 = self.conv1(self.attend_1(x,x_t,y_t))
        #x1 = self.attend_1(self.conv1(x),x_t,y_t)
        #x2 = self.w1(x)
        # x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
        # x2 = self.w1(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        x2 = self.w1(x)

        #x3, q, k = self.attend_1(x, q, k)
        x3, q, k = self.attend_1(x, x + y_t, x + x_t)
        #x3, q, k = self.attend_1(x, y_t, x_t)
        #x3, q, k = self.attend_1(x, q + y_t, k + x_t)
        x = x1 + x2 + x3
        x = F.gelu(x)


        x1 = self.conv2(x)
        #x1 = self.conv2(self.attend_2(x,x_t,y_t))
        #x1 = self.attend_2(self.conv2(x),x_t,y_t)
        #x2 = self.w2(x)
        # x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
        # x2 = self.w2(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        x2 = self.w2(x)

        #x3, q, k = self.attend_2(x, q, k)
        x3, q, k = self.attend_2(x, x + y_t, x + x_t)
        #x3, q, k = self.attend_2(x, y_t, x_t)
        #x3, q, k = self.attend_2(x, q + y_t, k + x_t)
        x = x1 + x2 + x3
        x = F.gelu(x)

        x1 = self.conv3(x)
        #x1 = self.conv3(self.attend_3(x,x_t,y_t))
        #x1 = self.attend_3(self.conv3(x),x_t,y_t)
        #x2 = self.w3(x)
        # x_ = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
        # x2 = self.w3(x_.permute(0,2,1)).permute(0,2,1).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        x2 = self.w3(x)

        #x3, q, k = self.attend_3(x, q, k)
        x3, q, k = self.attend_3(x, x + y_t, x + x_t)
        #x3, q, k = self.attend_3(x, y_t, x_t)
        #x3, q, k = self.attend_3(x, q + y_t, k + x_t)
        x = x1 + x2 + x3


        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        #print(x.shape)
        return x.squeeze(-1)

    def get_grid(self, shape, device):
        batchsize, size_x, = shape[0], shape[1]
        #gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)












































# def load_model(args, model_dict, device, multi_gpu=False, **kwargs):

#     #model_type ="FNO"

#     if args.model_type == "FNO":
#         if args.model_mode.startswith("constant_dt"):
#             model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps }
    
#         elif args.model_mode.startswith("variable_dt"):
#             if args.model_input_operation.startswith("add"):
#                 model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps }
#             elif args.model_input_operation.startswith("concat"):
#                 model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": 2*args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps }        
#         else:
#             raise TypeError("Specify the input shape type")
    
#         model = FNO1d(
#             model_hyperparameters["modes"],
#             model_hyperparameters["width"],
#             model_hyperparameters["input_size"],
#             model_hyperparameters["output_size"]
#             ).to(device)
        
#     elif args.model_type == "FNO1d_t":
#         if args.model_input_operation.startswith("concat"):
#             model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": 2*args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps, "x_res": args.x_res }   
        
#         model = FNO1d_t(
#             model_hyperparameters["modes"],
#             model_hyperparameters["width"],
#             model_hyperparameters["input_size"],
#             model_hyperparameters["output_size"],
#             model_hyperparameters["x_res"],

#             ).to(device)

    
    
    
#     elif args.model_type == "FNO1d_tparameter":
#         if args.model_input_operation.startswith("t_embed_attend"):
#             model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps, "t_res": args.total_t_range, "x_res": args.x_res }
        
#         model = FNO1d_tparameter(
#             model_hyperparameters["modes"],
#             model_hyperparameters["width"],
#             model_hyperparameters["input_size"],
#             model_hyperparameters["output_size"],
#             model_hyperparameters["t_res"],
#             model_hyperparameters["x_res"],

#             ).to(device)


#     p.print("Loading Model last state")
#     #model.load_state_dict(model_dict["state_dict"])
#     model.load_state_dict(model_dict)
#     model.to(device)
#     return model


# def get_model(args, device, **kwargs):
    
#     if args.model_type == "FNO":
#         if args.model_mode.startswith("constant_dt"):
#             if args.dataset_name.endswith("E1"):
#                 model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps }
#             elif args.dataset_name.endswith("E2"):
#                 model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps + 3, "output_size": args.output_time_stamps }


#         elif args.model_mode.startswith("variable_dt"):
#             if args.model_input_operation.startswith("add"):
#                 model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps }
#             elif args.model_input_operation.startswith("concat"):
#                 model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": 2*args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps }
#         else:
#             raise TypeError("Specify the input shape type")
    
#         model = FNO1d(
#             model_hyperparameters["modes"],
#             model_hyperparameters["width"],
#             model_hyperparameters["input_size"],
#             model_hyperparameters["output_size"]
#             ).to(device)


#     elif args.model_type == "FNO1d_t":
#         if args.model_input_operation.startswith("concat"):
#             model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": 2*args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps, "x_res": args.x_res }   
        
#         model = FNO1d_t(
#             model_hyperparameters["modes"],
#             model_hyperparameters["width"],
#             model_hyperparameters["input_size"],
#             model_hyperparameters["output_size"],
#             model_hyperparameters["x_res"],

#             ).to(device)
        

#     elif args.model_type == "FNO1d_tparameter":
#         if args.model_input_operation.startswith("t_embed_attend"):
#             model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps, "t_res": args.total_t_range, "x_res": args.x_res }
#         #import pdb; pdb.set_trace()
#         model = FNO1d_tparameter(
#             model_hyperparameters["modes"],
#             model_hyperparameters["width"],
#             model_hyperparameters["input_size"],
#             model_hyperparameters["output_size"],
#             model_hyperparameters["t_res"],
#             model_hyperparameters["x_res"],

#             ).to(device)
        
#     if args.model_initialise_type == "xavier_uniform":
#         initialize_weights_xavier_uniform(model)
#         p.print("Model with xavier_uniform")
#     else:
#         p.print("Model with Random Intitialisation")

#     return model.float()

    