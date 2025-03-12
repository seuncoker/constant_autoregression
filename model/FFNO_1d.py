import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm
import math
import copy

from constant_autoregression.util import Printer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Normalizer(nn.Module):
#     def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8):
#         super().__init__()
#         self.max_accumulations = max_accumulations
#         # self.register_buffer('count', torch.tensor(0.0))
#         # self.register_buffer('n_accumulations', torch.tensor(0.0))
#         # self.register_buffer('sum', torch.full(size, 0.0))
#         # self.register_buffer('sum_squared', torch.full(size, 0.0))
#         # self.register_buffer('one', torch.tensor(1.0))
#         # self.register_buffer('std_epsilon', torch.full(size, std_epsilon))
#         self.count =  torch.tensor(0.0, requires_grad=False).to(device)
#         self.n_accumulations = torch.tensor(0.0, requires_grad=False).to(device)
#         self.sum = torch.full(size, 0.0, requires_grad=False).to(device)
#         self.sum_squared = torch.full(size, 0.0, requires_grad=False).to(device)
#         self.one =  torch.tensor(1.0, requires_grad=False).to(device)
#         self.std_epsilon =  torch.full(size, std_epsilon, requires_grad=False).to(device)

#         self.dim_sizes = None

#     def _accumulate(self, x):
#         x_count = x.shape[0]
#         x_sum = x.sum(dim=0)
#         x_sum_squared = (x**2).sum(dim=0)

#         self.sum += x_sum
#         self.sum_squared += x_sum_squared
#         self.count += x_count
#         self.n_accumulations += 1

#     def _pool_dims(self, x):
#         _, *dim_sizes, _ = x.shape
#         self.dim_sizes = dim_sizes
#         if self.dim_sizes:
#             x = rearrange(x, 'b ... h -> (b ...) h')

#         return x

#     def _unpool_dims(self, x):
#         if len(self.dim_sizes) == 1:
#             x = rearrange(x, '(b m) h -> b m h', m=self.dim_sizes[0])
#         elif len(self.dim_sizes) == 2:
#             m, n = self.dim_sizes
#             x = rearrange(x, '(b m n) h -> b m n h', m=m, n=n)
#         return x

#     def forward(self, x):
#         x_in = self._pool_dims(x)
#         # x.shape == [batch_size, latent_dim]

#         if self.training and self.n_accumulations < self.max_accumulations:
#             self._accumulate(x_in)

#         x_out = (x_in - self.mean) / self.std
#         x_out1 = self._unpool_dims(x_out)

#         return x_out1

#     def inverse(self, x, channel=None):
#         x_in = self._pool_dims(x)

#         if channel is None:
#             x_out = x_in * self.std + self.mean
#         else:
#             x_out = x_in * self.std[channel] + self.mean[channel]

#         x_out1 = self._unpool_dims(x_out)

#         return x_out1

#     @property
#     def mean(self):
#         safe_count = max(self.count, self.one)
#         mean_value = self.sum / safe_count
#         return mean_value

#     @property
#     def std(self):
#         safe_count = max(self.count, self.one)
#         std = torch.sqrt(self.sum_squared / safe_count - self.mean**2)
#         std_value = torch.maximum(std, self.std_epsilon)
#         return std_value






class Normalizer(nn.Module):
    def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8):
        super().__init__()
        self.max_accumulations = max_accumulations
        self.register_buffer('count', torch.tensor(0.0))
        self.register_buffer('n_accumulations', torch.tensor(0.0))
        self.register_buffer('sum', torch.full(size, 0.0))
        self.register_buffer('sum_squared', torch.full(size, 0.0))
        self.register_buffer('one', torch.tensor(1.0))
        self.register_buffer('std_epsilon', torch.full(size, std_epsilon))
        self.dim_sizes = None

    def _accumulate(self, x):
        x_count = x.shape[0]
        x_sum = x.sum(dim=0)
        x_sum_squared = (x**2).sum(dim=0)

        self.sum += x_sum
        self.sum_squared += x_sum_squared
        self.count += x_count
        self.n_accumulations += 1

    def _pool_dims(self, x):
        _, *dim_sizes, _ = x.shape
        self.dim_sizes = dim_sizes
        if self.dim_sizes:
            x = rearrange(x, 'b ... h -> (b ...) h')

        return x

    def _unpool_dims(self, x):
        if len(self.dim_sizes) == 1:
            x = rearrange(x, '(b m) h -> b m h', m=self.dim_sizes[0])
        elif len(self.dim_sizes) == 2:
            m, n = self.dim_sizes
            x = rearrange(x, '(b m n) h -> b m n h', m=m, n=n)
        return x

    def forward(self, x):
        x = self._pool_dims(x)
        # x.shape == [batch_size, latent_dim]

        if self.training and self.n_accumulations < self.max_accumulations:
            self._accumulate(x)

        x = (x - self.mean) / self.std
        x = self._unpool_dims(x)

        return x

    def inverse(self, x, channel=None):
        x = self._pool_dims(x)

        if channel is None:
            x = x * self.std + self.mean
        else:
            x = x * self.std[channel] + self.mean[channel]

        x = self._unpool_dims(x)

        return x

    @property
    def mean(self):
        safe_count = max(self.count, self.one)
        return self.sum / safe_count

    @property
    def std(self):
        safe_count = max(self.count, self.one)
        std = torch.sqrt(self.sum_squared / safe_count - self.mean**2)
        return torch.maximum(std, self.std_epsilon)
    




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
    




class FeedForward(nn.Module):
    def __init__(self, dim, factor, ff_weight_norm, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                WNLinear(in_dim, out_dim, wnorm=ff_weight_norm),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    


class WNLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, wnorm=False):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype)
        if wnorm:
            weight_norm(self)

        self._fix_weight_norm_deepcopy()

    def _fix_weight_norm_deepcopy(self):
        # Fix bug where deepcopy doesn't work with weightnorm.
        # Taken from https://github.com/pytorch/pytorch/issues/28594#issuecomment-679534348
        orig_deepcopy = getattr(self, '__deepcopy__', None)

        def __deepcopy__(self, memo):
            # save and delete all weightnorm weights on self
            weights = {}
            for hook in self._forward_pre_hooks.values():
                if isinstance(hook, WeightNorm):
                    weights[hook.name] = getattr(self, hook.name)
                    delattr(self, hook.name)
            # remove this deepcopy method, restoring the object's original one if necessary
            __deepcopy__ = self.__deepcopy__
            if orig_deepcopy:
                self.__deepcopy__ = orig_deepcopy
            else:
                del self.__deepcopy__
            # actually do the copy
            result = copy.deepcopy(self)
            # restore weights and method on self
            for name, value in weights.items():
                setattr(self, name, value)
            self.__deepcopy__ = __deepcopy__
            return result
        # bind __deepcopy__ to the weightnorm'd layer
        self.__deepcopy__ = __deepcopy__.__get__(self, self.__class__)


class SpectralConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout, mode):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.mode = mode
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            #self.fourier_weight = nn.ParameterList([])
            # for _ in range(2):
            #     weight = torch.Tensor(in_dim, out_dim, n_modes, 2)
            #     param = nn.Parameter(weight)
            #     nn.init.xavier_normal_(param)
            #     self.fourier_weight.append(param)
            fourier_weight = [nn.Parameter(torch.FloatTensor(
                in_dim, out_dim, n_modes, 2)) for _ in range(2)]
            self.fourier_weight = nn.ParameterList(fourier_weight)
            for param in self.fourier_weight:
                nn.init.xavier_normal_(param, gain=1/(in_dim*out_dim))

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b x i -> b i x')
        # x.shape == [batch_size, in_dim, grid_size]

        B, I, N = x.shape

        # # # Dimesion X # # #
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_ft.new_zeros(B, I, N // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]
        if self.mode == 'full':
            out_ft[:, :, :self.n_modes] = torch.einsum(
                "bix,iox->box",
                x_ft[:, :, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]))
            
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.n_modes] = x_ft[:, :, :self.n_modes]

        x = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]



        x = rearrange(x, 'b i x -> b x i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x


class F_FNO_1D(nn.Module):
    def __init__(self, modes, width, input_dim=12, output_dim=1, dropout=0.0, in_dropout=0.0,
                 n_layers=4, share_weight: bool = False,
                 share_fork=False, factor=2,
                 ff_weight_norm=False, n_ff_layers=2,
                 gain=1, layer_norm=False, use_fork=False, mode='full'):
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers
        self.use_fork = use_fork
        #self.normalizer = Normalizer_1D()
        #self.normalizer = Normalizer([input_dim], )

        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            if use_fork:
                self.forecast_ff = FeedForward(
                    width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
            self.backcast_ff = FeedForward(
                width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.fourier_weight = None
        if share_weight:
            # self.fourier_weight = nn.ParameterList([])
            # for _ in range(2):
            #     weight = torch.tensor((width, width, modes, 2), dtype=float)
            #     param = nn.Parameter(weight)
            #     nn.init.xavier_normal_(param, gain=gain)
            #     self.fourier_weight.append(param)

            fourier_weight = [nn.Parameter(torch.FloatTensor(
                width, width, modes, 2)) for _ in range(2)]
            self.fourier_weight = nn.ParameterList(fourier_weight)
            for param in self.fourier_weight:
                nn.init.xavier_normal_(param, gain=1/(width*width))

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv1d(in_dim=width,
                                                       out_dim=width,
                                                       n_modes=modes,
                                                       forecast_ff=self.forecast_ff,
                                                       backcast_ff=self.backcast_ff,
                                                       fourier_weight=self.fourier_weight,
                                                       factor=factor,
                                                       ff_weight_norm=ff_weight_norm,
                                                       n_ff_layers=n_ff_layers,
                                                       layer_norm=layer_norm,
                                                       use_fork=use_fork,
                                                       dropout=dropout,
                                                       mode=mode))

        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, self.output_dim, wnorm=ff_weight_norm))

        

    def forward(self, x, **kwargs):
        # x.shape == [n_batches, *dim_sizes, input_size]
        forecast = 0
        #x = self.normalizer(x)
        x = self.in_proj(x)
        x = self.drop(x)
        forecast_list = []
        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b, f = layer(x)

            if self.use_fork:
                f_out = self.out(f)
                forecast = forecast + f_out
                forecast_list.append(f_out)

            x = x + b

        if not self.use_fork:
            forecast = self.out(b)

        #forecast = self.normalizer.inverse(forecast)
        return forecast
    



# class FNOFactorized2DBlock(nn.Module):
#     def __init__(self, modes, width, input_dim=12, dropout=0.0, in_dropout=0.0,
#                  n_layers=4, share_weight: bool = False,
#                  share_fork=False, factor=2,
#                  ff_weight_norm=False, n_ff_layers=2,
#                  gain=1, layer_norm=False, use_fork=False, mode='full'):
#         super().__init__()
#         self.modes = modes
#         self.width = width
#         self.input_dim = input_dim
#         self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
#         self.drop = nn.Dropout(in_dropout)
#         self.n_layers = n_layers
#         self.use_fork = use_fork

#         self.forecast_ff = self.backcast_ff = None
#         if share_fork:
#             if use_fork:
#                 self.forecast_ff = FeedForward(
#                     width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
#             self.backcast_ff = FeedForward(
#                 width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

#         self.fourier_weight = None
#         if share_weight:
#             # self.fourier_weight = nn.ParameterList([])
#             # for _ in range(2):
#             #     weight = torch.tensor((width, width, modes, 2), dtype=float)
#             #     param = nn.Parameter(weight)

#             #     nn.init.xavier_normal_(param, gain=gain)
#             #     self.fourier_weight.append(param)


#             fourier_weight = [nn.Parameter(torch.FloatTensor(
#                 width, width, modes, 2)) for _ in range(2)]
#             self.fourier_weight = nn.ParameterList(fourier_weight)
#             for param in self.fourier_weight:
#                 nn.init.xavier_normal_(param, gain=1/(width*width))


#         self.spectral_layers = nn.ModuleList([])
#         for _ in range(n_layers):
#             self.spectral_layers.append(SpectralConv1d(in_dim=width,
#                                                        out_dim=width,
#                                                        n_modes=modes,
#                                                        forecast_ff=self.forecast_ff,
#                                                        backcast_ff=self.backcast_ff,
#                                                        fourier_weight=self.fourier_weight,
#                                                        factor=factor,
#                                                        ff_weight_norm=ff_weight_norm,
#                                                        n_ff_layers=n_ff_layers,
#                                                        layer_norm=layer_norm,
#                                                        use_fork=use_fork,
#                                                        dropout=dropout,
#                                                        mode=mode))

#         self.out = nn.Sequential(
#             WNLinear(self.width, 128, wnorm=ff_weight_norm),
#             WNLinear(128, 1, wnorm=ff_weight_norm))

#     def forward(self, x, **kwargs):
#         # x.shape == [n_batches, *dim_sizes, input_size]
#         forecast = 0
#         x = self.in_proj(x)
#         x = self.drop(x)
#         forecast_list = []
#         for i in range(self.n_layers):
#             layer = self.spectral_layers[i]
#             b, f = layer(x)

#             if self.use_fork:
#                 f_out = self.out(f)
#                 forecast = forecast + f_out
#                 forecast_list.append(f_out)

#             x = x + b

#         if not self.use_fork:
#             forecast = self.out(b)

#         return {
#             'forecast': forecast,
#             'forecast_list': forecast_list,
#         }