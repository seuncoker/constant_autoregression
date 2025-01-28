import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
import copy
import operator
from functools import reduce
from functools import partial
import numpy as np






class SpectralConv1d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1,modes1 = None):
        super(SpectralConv1d_Uno, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        Ratio of grid size of the input and the output implecitely 
        set the expansion or contraction farctor along each dimension of the domain.
        modes1 = Number of fourier modes to consider for the integral operator.
                Number of modes must be compatibale with the input grid size 
                and desired output grid size.
                i.e., modes1 <= min( dim1/2, input_dim1/2). 
                Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension
        """
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1 #output dimensions
        if modes1 is not None:
            self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        else:
            self.modes1 = dim1//2

        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, dim1 = None):
        """
        input shape = (batch, in_codim, input_dim1)
        output shape = (batch, out_codim, dim1)
        """
        #import pdb; pdb.set_trace()
        if dim1 is not None:
            self.dim1 = dim1
        batchsize = x.shape[0]
        
        # print("conv...")
        # print("x_in ->", x.shape)
        x_ft = torch.fft.rfft(x, norm = 'forward')
        #print("x_ft - >", x_ft.shape)
        # Multiply relevant Fourier modes
        #print("Out (xft, weight) ->", x_ft[:, :, :self.modes1].shape,  self.weights1.shape )
        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=self.dim1, norm = 'forward')
        #print("x ->", x.shape)
        return x

class pointwise_op_1D(nn.Module):
    """
    All variables are consistent with the SpectralConv1d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1):
        super(pointwise_op_1D,self).__init__()
        self.conv = nn.Conv1d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)

    def forward(self,x, dim1 = None):
        #import pdb; pdb.set_trace()
        if dim1 is None:
            dim1 = self.dim1
        x_out = self.conv(x)

        #x_out = torch.nn.functional.interpolate(x_out, size = dim1,mode = 'linear',align_corners=True, antialias = True)
        x_out = torch.nn.functional.interpolate(x_out, size = dim1,mode = 'linear',align_corners=True)
        return x_out



class OperatorBlock_1D(nn.Module):
    """
    Normalize = if true performs InstanceNorm1d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv1d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1,modes1, Normalize = True,Non_Lin = True):
        super(OperatorBlock_1D,self).__init__()
        self.conv = SpectralConv1d_Uno(in_codim, out_codim, dim1,modes1)
        self.w = pointwise_op_1D(in_codim, out_codim, dim1)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm1d(int(out_codim),affine=True)


    def forward(self,x, dim1 = None):
        """
        input shape = (batch, in_codim, input_dim1)
        output shape = (batch, out_codim, dim1)
        """
        # #import pdb; pdb.set_trace()
        # print("\n")
        # print("x_in ->", x.shape)
        x1_out = self.conv(x,dim1)

        #print("x1_out ->", x1_out.shape)

        x2_out = self.w(x,dim1)

        #print("x2_out ->", x2_out.shape)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out



# UNO model 
# it has less aggressive scaling factors for domains and co-domains.    
class UNO_1D(nn.Module):
    def __init__(self,in_width, out_dim, width, pad = 0, factor = 3/4):
        super(UNO_1D, self).__init__()


        self.in_width = in_width + 1 # input channel
        self.width = width 
        self.factor = factor
        self.padding = pad
        self.out_dim = out_dim

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

   
        self.L0 = OperatorBlock_1D(self.width, 2*factor*self.width,48, 22)

        self.L1 = OperatorBlock_1D(2*factor*self.width, 4*factor*self.width, 32, 14)

        self.L2 = OperatorBlock_1D(4*factor*self.width, 8*factor*self.width, 16, 6,)
        
        self.L3 = OperatorBlock_1D(8*factor*self.width, 8*factor*self.width, 16, 6)
        
        self.L4 = OperatorBlock_1D(8*factor*self.width, 4*factor*self.width, 32, 6)

        self.L5 = OperatorBlock_1D(8*factor*self.width, 2*factor*self.width, 48,14)

        self.L6 = OperatorBlock_1D(4*factor*self.width, self.width, 64, 22) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, self.out_dim)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        grid = self.get_grid(x.shape, x.device)
        # print("x ->",x.shape)
        # print("grid ->", grid.shape)
        x = torch.cat((x, grid), dim=-1)

        #print("x + grid ->",x.shape)


        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        #print("x_fc ->",x_fc.shape)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        #print("x_fc0 ->",x_fc0.shape)


        x_fc0 = x_fc0.permute(0, 2, 1)
        
        #print("x_fc0 ->",x_fc0.shape)
        
        x_fc0 = F.pad(x_fc0, [self.padding,self.padding, self.padding,self.padding])
        
        #print("x_fc0 pad ->",x_fc0.shape)

        D1 = x_fc0.shape[-1]
        
        #print("cofactor ->", D1, self.factor, int(D1*self.factor))
        x_c0 = self.L0(x_fc0,int(D1*self.factor))
        #print("x_c0 ->",x_c0.shape)

        x_c1 = self.L1(x_c0 ,D1//2)
        #print("x_c1 ->",x_c1.shape)

        x_c2 = self.L2(x_c1 ,D1//4)
        #print("x_c2 ->",x_c2.shape)

        x_c3 = self.L3(x_c2,D1//4)
        #print("x_c3 ->",x_c3.shape)


        x_c4 = self.L4(x_c3,D1//2)
        #print("x_c4 ->",x_c4.shape)

        x_c4 = torch.cat([x_c4, x_c1], dim=1)

        x_c5 = self.L5(x_c4,int(D1*self.factor))
        #print("x_c5 ->",x_c5.shape)
        #print("x_c0 ->", x_c0.shape)
        x_c5 = torch.cat([x_c5, x_c0], dim=1)

        x_c6 = self.L6(x_c5,D1)
        #print("x_c6 ->",x_c6.shape)
        #print(print("x_fc0 ->", x_fc0.shape))
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding!=0:
            x_c6 = x_c6[..., :-self.padding, :-self.padding]

        x_c6 = x_c6.permute(0, 2, 1)
        
        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)
        
        x_out = self.fc2(x_fc1)
        #print("x_out ->", x_out.shape)
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2*np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        # gridy = torch.tensor(np.linspace(0, 2*np.pi, size_y), dtype=torch.float)
        # gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        #return torch.cat((torch.sin(gridx),torch.sin(gridy),torch.cos(gridx),torch.cos(gridy)), dim=-1).to(device)
        return (torch.sin(gridx) + torch.cos(gridx) ).to(device)















# class SpectralConv1d_Uno(nn.Module):
#     def __init__(self, in_codim, out_codim, dim1,modes1 = None):
#         super(SpectralConv1d_Uno, self).__init__()

#         """
#         1D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
#         dim1 = Default output grid size along x (or 1st dimension of output domain) 
#         Ratio of grid size of the input and the output implecitely 
#         set the expansion or contraction farctor along each dimension of the domain.
#         modes1 = Number of fourier modes to consider for the integral operator.
#                 Number of modes must be compatibale with the input grid size 
#                 and desired output grid size.
#                 i.e., modes1 <= min( dim1/2, input_dim1/2). 
#                 Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
#         in_codim = Input co-domian dimension
#         out_codim = output co-domain dimension
#         """
#         in_codim = int(in_codim)
#         out_codim = int(out_codim)
#         self.in_channels = in_codim
#         self.out_channels = out_codim
#         self.dim1 = dim1 #output dimensions
#         if modes1 is not None:
#             self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
#         else:
#             self.modes1 = dim1//2

#         self.scale = (1 / (2*in_codim))**(1.0/2.0)
#         self.weights1 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, dtype=torch.cfloat))

#     # Complex multiplication
#     def compl_mul1d(self, input, weights):
#         # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
#         return torch.einsum("bix,iox->box", input, weights)

#     def forward(self, x, dim1 = None):
#         """
#         input shape = (batch, in_codim, input_dim1)
#         output shape = (batch, out_codim, dim1)
#         """
#         import pdb; pdb.set_trace()
#         if dim1 is not None:
#             self.dim1 = dim1
#         batchsize = x.shape[0]

#         x_ft = torch.fft.rfft(x, norm = 'forward')

#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1//2 + 1 , dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

#         #Return to physical space
#         x = torch.fft.irfft(out_ft, n=self.dim1, norm = 'forward')
#         return x

# class pointwise_op_1D(nn.Module):
#     """
#     All variables are consistent with the SpectralConv1d_Uno class.
#     """
#     def __init__(self, in_codim, out_codim,dim1):
#         super(pointwise_op_1D,self).__init__()
#         self.conv = nn.Conv1d(int(in_codim), int(out_codim), 1)
#         self.dim1 = int(dim1)

#     def forward(self,x, dim1 = None):
#         import pdb; pdb.set_trace()
#         if dim1 is None:
#             dim1 = self.dim1
#         x_out = self.conv(x)

#         x_out = torch.nn.functional.interpolate(x_out, size = dim1,mode = 'linear',align_corners=True, antialias= True)
#         return x_out



# class OperatorBlock_1D(nn.Module):
#     """
#     Normalize = if true performs InstanceNorm1d on the output.
#     Non_Lin = if true, applies point wise nonlinearity.
#     All other variables are consistent with the SpectralConv1d_Uno class.
#     """
#     def __init__(self, in_codim, out_codim,dim1,modes1, Normalize = True,Non_Lin = True):
#         super(OperatorBlock_1D,self).__init__()
#         self.conv = SpectralConv1d_Uno(in_codim, out_codim, dim1,modes1)
#         self.w = pointwise_op_1D(in_codim, out_codim, dim1)
#         self.normalize = Normalize
#         self.non_lin = Non_Lin
#         if Normalize:
#             self.normalize_layer = torch.nn.InstanceNorm1d(int(out_codim),affine=True)


#     def forward(self,x, dim1 = None):
#         """
#         input shape = (batch, in_codim, input_dim1)
#         output shape = (batch, out_codim, dim1)
#         """
#         import pdb; pdb.set_trace()
#         x1_out = self.conv(x,dim1)
#         x2_out = self.w(x,dim1)
#         x_out = x1_out + x2_out
#         if self.normalize:
#             x_out = self.normalize_layer(x_out)
#         if self.non_lin:
#             x_out = F.gelu(x_out)
#         return x_out



# # UNO model 
# # it has less aggressive scaling factors for domains and co-domains.
# # ####    
# class UNO_1D(nn.Module):
#     def __init__(self,in_width, width,pad = 0, factor = 3/4):
#         super(UNO_1D, self).__init__()


#         self.in_width = in_width # input channel
#         self.width = width 
#         self.factor = factor
#         self.padding = pad  

#         self.fc0 = nn.Linear(self.width, self.width) # input channel is 3: (a(x), x)

#         self.L0 = OperatorBlock_1D(self.width, 2*factor*self.width,48, 22)

#         self.L1 = OperatorBlock_1D(2*factor*self.width, 4*factor*self.width, 32, 14)

#         self.L2 = OperatorBlock_1D(4*factor*self.width, 8*factor*self.width, 16, 6,)
        
#         self.L3 = OperatorBlock_1D(8*factor*self.width, 8*factor*self.width, 16, 6)
        
#         self.L4 = OperatorBlock_1D(8*factor*self.width, 4*factor*self.width, 32, 6)

#         self.L5 = OperatorBlock_1D(8*factor*self.width, 2*factor*self.width, 48,14)

#         self.L6 = OperatorBlock_1D(4*factor*self.width, self.width, 64, 22) # will be reshaped

#         self.fc1 = nn.Linear(2*self.width, 4*self.width)
#         self.fc2 = nn.Linear(4*self.width, 1)

#     def forward(self, x):
#         import pdb; pdb.set_trace()
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)
         
#         x_fc0 = self.fc0(x)

#         x_fc0 = x_fc0.permute(0, 2, 1)
        
        
#         x_fc0 = F.pad(x_fc0, [self.padding,self.padding, self.padding,self.padding])
        
#         D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        
#         x_c0 = self.L0(x_fc0,int(D1*self.factor),int(D2*self.factor))
#         x_c1 = self.L1(x_c0 ,D1//2,D2//2)

#         x_c2 = self.L2(x_c1 ,D1//4,D2//4)        
#         x_c3 = self.L3(x_c2,D1//4,D2//4)
#         x_c4 = self.L4(x_c3,D1//2,D2//2)
#         x_c4 = torch.cat([x_c4, x_c1], dim=1)
#         x_c5 = self.L5(x_c4,int(D1*self.factor),int(D2*self.factor))
#         x_c5 = torch.cat([x_c5, x_c0], dim=1)
#         x_c6 = self.L6(x_c5,D1,D2)
#         x_c6 = torch.cat([x_c6, x_fc0], dim=1)

#         if self.padding!=0:
#             x_c6 = x_c6[..., :-self.padding, :-self.padding]

#         x_c6 = x_c6.permute(0, 2, 3, 1)
        
#         x_fc1 = self.fc1(x_c6)
#         x_fc1 = F.gelu(x_fc1)
        
#         x_out = self.fc2(x_fc1)
        
#         return x_out
    
#     def get_grid(self, shape, device):
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.tensor(np.linspace(0, 2*np.pi, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(np.linspace(0, 2*np.pi, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((torch.sin(gridx),torch.sin(gridy),torch.cos(gridx),torch.cos(gridy)), dim=-1).to(device)
