import numpy as np
import torch
import torch.nn as nn
import csv
import h5py
from torch.utils.data import Dataset
#from termcolor import colored
import sys, os
from datetime import datetime
from typing import Tuple
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

from constant_autoregression.argparser import arg_parse 
from util import set_seed

args = arg_parse()
set_seed(args.seed)


betaA_3 = np.sqrt(13 / 12) * np.array([[[1, -2, 1, 0, 0]],
                                             [[0, 1, -2, 1, 0]],
                                             [[0, 0, 1, -2, 1]]])

betaB_3 = (1 / 2) * np.array([[[1, -4, 3, 0, 0]],
                                    [[0, 1, 0, -1, 0]],
                                    [[0, 0, 3, -4, 1]]])

gamma_3 = np.array([[[1],[6],[3]]]) / 10

stencils_3 = (1 / 6) * np.array([[[2,  -7,  11,  0,  0]],
                                    [[ 0,  -1,  5, 2,  0]],
                                    [[ 0,  0,  2, 5,  -1]]])


betaA_all = {
          3 : betaA_3
        }

betaB_all = {
          3 : betaB_3
        }

gamma_all = {
          3 : gamma_3
        }

stencils_all =  {
          3 : stencils_3
        }
###########################################################################
# Coefficients of central differences for 4th order accuracy for first and second derivative and
# 2nd order accuracy for third and fourth derivative (for simplicity)
# Coefficients taken from: https://en.wikipedia.org/wiki/Finite_difference_coefficient
derivative_1 = np.array([[[1/12, -2/3, 0, 2/3, -1/12]]])
derivative_2 = np.array([[[-1/12, 4/3, -5/2, 4/3, -1/12]]])
derivative_3 = np.array([[[-1/2, 1, 0, -1, 1/2]]])
#derivative_3 = np.array([[[-7/240, 3/10, -169/120, 61/30, 0, -61/30, 169/120, -3/10, 7/240]]])
derivative_4 = np.array([[[1., -4., 6., -4., 1.]]])

FDM_derivatives = {
        1 : derivative_1,
        2 : derivative_2,
        3 : derivative_3,
        4:  derivative_4
}


class PDE(nn.Module):
    """Generic PDE template"""
    def __init__(self):
        # Data params for grid and initial conditions
        super().__init__()
        pass

    def __repr__(self):
        return "PDE"

    def FDM_reconstruction(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """A finite differences method template"""
        pass

    def FVM_reconstruction(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """A finite volumes method template"""
        pass

    def WENO_reconstruction(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """A WENO reconstruction template"""
        pass


# class HDF5Dataset(Dataset):
#     """Load samples of an PDE Dataset, get items according to PDE"""

#     def __init__(self,
#                  path: str,
#                  pde: PDE,
#                  mode: str,
#                  base_resolution: list=None,
#                  super_resolution: list=None,
#                  load_all: bool=False,
#                  uniform_sample: int=-1,
#                  is_return_super = False,
#                 ) -> None:
#         """Initialize the dataset object
#         Args:
#             path: path to dataset
#             pde: string of PDE ('CE' or 'WE')
#             mode: [train, valid, test]
#             base_resolution: base resolution of the dataset [nt, nx]
#             super_resolution: super resolution of the dataset [nt, nx]
#             load_all: load all the data into memory
#         Returns:
#             None
#         """
#         super().__init__()
#         f = h5py.File(path, 'r')
#         self.mode = mode
#         #self.pde = pde
#         self.dtype = torch.float64
#         self.data = f[self.mode]
#         self.base_resolution = (250, 100) if base_resolution is None else base_resolution
#         self.super_resolution = (250, 200) if super_resolution is None else super_resolution
#         self.uniform_sample = uniform_sample
#         self.dataset_base = f'pde_{self.base_resolution[0]}-{self.base_resolution[1]}'
#         self.dataset_super = f'pde_{self.super_resolution[0]}-{self.super_resolution[1]}'
#         self.is_return_super = is_return_super

#         if self.base_resolution[1] != 34:
#             ratio_nt = self.data[self.dataset_super].shape[1] / self.data[self.dataset_base].shape[1]
#             ratio_nx = self.data[self.dataset_super].shape[2] / self.data[self.dataset_base].shape[2]
#         else:
#             ratio_nt = 1.0
#             ratio_nx = int(200 / 33)
#         # assert (ratio_nt.is_integer())
#         # assert (ratio_nx.is_integer())
#         self.ratio_nt = int(ratio_nt)
#         self.ratio_nx = int(ratio_nx)

#         if self.base_resolution[1] != 34:
#             self.nt = self.data[self.dataset_base].attrs['nt']
#             self.dt = self.data[self.dataset_base].attrs['dt']
#             self.dx = self.data[self.dataset_base].attrs['dx']
#             self.x = self.data[self.dataset_base].attrs['x']
#             self.tmin = self.data[self.dataset_base].attrs['tmin']
#             self.tmax = self.data[self.dataset_base].attrs['tmax']
#         else:
#             self.nt = 250
#             self.dt = self.data['pde_250-50'].attrs['dt']
#             self.dx = 0.16 * 3
#             self.x = self.data['pde_250-100'].attrs['x'][::3]
#             self.tmin = self.data['pde_250-50'].attrs['tmin']
#             self.tmax = self.data['pde_250-50'].attrs['tmax']
#         self.x_ori = self.data['pde_250-100'].attrs['x']

#         if load_all:
#             data = {self.dataset_super: self.data[self.dataset_super][:]}
#             f.close()
#             self.data = data


#     def __len__(self):
#         return self.data[self.dataset_super].shape[0]

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
#         """
#         Get data item
#         Args:
#             idx (int): data index
#         Returns:
#             torch.Tensor: numerical baseline trajectory
#             torch.Tensor: downprojected high-resolution trajectory (used for training)
#             torch.Tensor: spatial coordinates
#             list: equation specific parameters
#         """
#         if(f'{self.pde}' == 'CE'):
#             # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
#             u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
#             left = u_super[..., -3:-1]
#             right = u_super[..., 1:3]
#             u_super_padded = torch.DoubleTensor(np.concatenate((left, u_super, right), -1))
#             weights = torch.DoubleTensor([[[[0.2]*5]]])
#             if self.uniform_sample == -1:
#                 u_super = F.conv1d(u_super_padded, weights, stride=(1, self.ratio_nx)).squeeze().numpy()
#             else:
#                 u_super = F.conv1d(u_super_padded, weights, stride=(1, 2)).squeeze().numpy()
#             x = self.x

#             # pdb.set_trace()
#             # Base resolution trajectories (numerical baseline) and equation specific parameters
#             if self.uniform_sample != -1:
#                 u_base = self.data[f'pde_{self.base_resolution[0]}-{100}'][idx][...,::self.uniform_sample]
#                 u_super_core = u_super[...,::self.uniform_sample]
#             else:
#                 u_base = self.data[self.dataset_base][idx]
#                 u_super_core = u_super
#             variables = {}
#             variables['alpha'] = self.data['alpha'][idx]
#             variables['beta'] = self.data['beta'][idx]
#             variables['gamma'] = self.data['gamma'][idx]

#             if self.is_return_super:
#                 return u_base, u_super_core, u_super, x, self.x_ori, variables
#             else:
#                 return u_base, u_super_core, x, variables

#         elif(f'{self.pde}' == 'WE'):
#             # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
#             # No padding is possible due to non-periodic boundary conditions
#             weights = torch.tensor([[[[1./self.ratio_nx]*self.ratio_nx]]])
#             u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
#             u_super = F.conv1d(torch.tensor(u_super), weights, stride=(1, self.ratio_nx)).squeeze().numpy()

#             # To match the downprojected trajectories, also coordinates need to be downprojected
#             x_super = torch.tensor(self.data[self.dataset_super].attrs['x'][None, None, None, :])
#             x = F.conv1d(x_super, weights, stride=(1, self.ratio_nx)).squeeze().numpy()

#             # Base resolution trajectories (numerical baseline) and equation specific parameters
#             u_base = self.data[self.dataset_base][idx]
#             variables = {}
#             variables['bc_left'] = self.data['bc_left'][idx]
#             variables['bc_right'] = self.data['bc_right'][idx]
#             variables['c'] = self.data['c'][idx]

#             return u_base, u_super, x, variables

#         else:
#             raise Exception("Wrong experiment")


class FDM():
    """
    FDM reconstruction
    """
    def __init__(self, pde, device: torch.cuda.device="cpu") -> None:
        """
        Initialize FDM reconstruction class
        Args:
            pde (PDE): PDE at hand
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        super().__init__()
        self.device = device
        self.pde = pde
        self.weights1 = torch.tensor(FDM_derivatives[1]).to(self.device)
        self.weights2 = torch.tensor(FDM_derivatives[2]).to(self.device)
        self.weights3 = torch.tensor(FDM_derivatives[3]).to(self.device)
        self.weights4 = torch.tensor(FDM_derivatives[4]).to(self.device)

    def pad(self, input: torch.Tensor) -> torch.Tensor:
        """
        Padding according to FDM derivatives for periodic boundary conditions
        Padding with size 2 is correct for 4th order accuracy for first and second derivative and
        for 2nd order accuracy for third and fourth derivative (for simplicity)
        """
        left = input[..., -3:-1]
        right = input[..., 1:3]
        padded_input = torch.cat([left, input, right], -1)
        return padded_input

    def first_derivative(self, input: torch.Tensor) -> torch.Tensor:
        """
        FDM method for first order derivative
        """
        return (1 / self.pde.dx) * F.conv1d(input, self.weights1)

    def second_derivative(self, input: torch.Tensor) -> torch.Tensor:
        """
            FDM method for second order derivative
        """
        return (1 / self.pde.dx)**2 * F.conv1d(input, self.weights2)

    def third_derivative(self, input: torch.Tensor) -> torch.Tensor:
        """
            FDM method for third order derivative
        """
        return (1 / self.pde.dx)**3 * F.conv1d(input, self.weights3)

    def fourth_derivative(self, input: torch.Tensor) -> torch.Tensor:
        """
            FDM method for fourth order derivative
        """
        return (1 / self.pde.dx)**4 * F.conv1d(input, self.weights4)


class WENO():
    """
    WENO5 reconstruction
    """
    def __init__(self, pde, order: int=3, device: torch.cuda.device="cpu") -> None:
        """
        Initialization of GPU compatible WENO5 method
        Args:
            pde (PDE): PDE at hand
            order (int): order of WENO coefficients (order 3 for WENO5 method)
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        super().__init__()

        self.pde = pde
        self.order = order
        self.epsilon = 1e-16
        self.device = device

        assert(self.order == 3) # as default for WENO5 scheme, higher orders are not implemented
        betaA = betaA_all[self.order]
        betaB = betaB_all[self.order]
        gamma = gamma_all[self.order]
        stencils = stencils_all[self.order]

        self.betaAm = torch.tensor(betaA).to(self.device)
        self.betaBm = torch.tensor(betaB).to(self.device)
        self.gamma = torch.tensor(gamma).to(self.device)
        self.stencils = torch.tensor(stencils).to(self.device)

    def pad(self, input: torch.Tensor) -> torch.Tensor:
        """
        Padding according to order of Weno scheme
        """
        left = input[..., -self.order:-1]
        right = input[..., 1:self.order]
        padded_input = torch.cat([left, input, right], -1)
        return padded_input

    def reconstruct_godunov(self, input: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct via Godunov flux
        Args:
            input (torch.Tensor): padded input
            dx (torch.Tensor): step size
        Returns:
            torch.Tensor: reconstructed Godunov flux
        """
        # reconstruct from the right
        rec_plus = self.reconstruct(torch.flip(input, [-1]))
        rec_plus = torch.flip(rec_plus, [-1])
        rec_plus = torch.roll(rec_plus, -1, -1)
        # reconstruct from the left
        rec_minus = self.reconstruct(input)

        switch = torch.ge(rec_plus, rec_minus).type(torch.float64)
        flux_plus = self.pde.flux(rec_plus)
        flux_minus = self.pde.flux(rec_minus)
        min_flux = torch.min(flux_minus, flux_plus)
        max_flux = torch.max(flux_minus, flux_plus)
        flux_out = switch * min_flux + (1 - switch) * max_flux
        flux_in = torch.roll(flux_out, 1, -1)
        flux_godunov = 1 / dx * (flux_out - flux_in)
        return flux_godunov


    def reconstruct_laxfriedrichs(self, input: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct via Lax-Friedrichs flux
        Args:
            input (torch.Tensor): padded input
            dx (torch.Tensor): step size
        Returns:
            torch.Tensor: reconstructed Lax-Friedrichs flux
        """
        f = self.pde.flux(input)
        alpha = torch.max(input, -1).values
        f_plus = f + alpha * input
        f_minus = f-alpha * input

        # construct flux from the left
        flux_plus = self.reconstruct(f_plus) / 2
        # construct flux from the right
        flux_minus = self.reconstruct(torch.flip(f_minus, [-1])) / 2
        flux_minus = torch.flip(flux_minus, [-1])
        flux_minus = torch.roll(flux_minus, -1, -1)
        # add fluxes
        flux_out = flux_plus + flux_minus
        flux_in = torch.roll(flux_out, 1, -1)
        flux_laxfriedrichs = 1 / dx * (flux_out - flux_in)
        return flux_laxfriedrichs


    def reconstruct(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Weno5 reconstruction
        '''

        b1 = F.conv1d(input, self.betaAm)
        b2 = F.conv1d(input, self.betaBm)
        beta = b1 * b1 + b2 * b2

        w_tilde = self.gamma / (self.epsilon + beta) ** 2
        w = (w_tilde / torch.sum(w_tilde, axis=1, keepdim=True)).view(-1, 1, 3, w_tilde.shape[-1])

        derivatives = F.conv1d(input, self.stencils).view(input.shape[0], -1, 3, w.shape[-1])
        output = torch.sum(w * derivatives, axis=2)

        return output




class CE(PDE):
    """
    Combined equation with Burgers and KdV as edge cases
    ut = -alpha*uux + beta*uxx + -gamma*uxxx = 0
    alpha = 6 for KdV and alpha = 1. for Burgers
    beta = nu for Burgers
    gamma = 1 for KdV
    alpha = 0, beta = nu, gamma = 0 for heat equation
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 L: float=None,
                 flux_splitting: str=None,
                 alpha: float=3.,
                 beta: float=0.,
                 gamma: float=1.,
                 device: torch.cuda.device = "cpu") -> None:
        """
        Args:
            tmin (float): starting time
            tmax (float): end time
            grid_size (list): grid points [nt, nx]
            L (float): periodicity
            flux_splitting (str): flux splitting used for WENO reconstruction (Godunov, Lax-Friedrichs)
            alpha (float): shock term
            beta (float): viscosity/diffusion parameter
            gamma (float): dispersive parameter
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        # Data params for grid and initial conditions
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 0.5 if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1
        self.lmax = 3
        # Number of different waves
        self.N = 5
        # Length of the spatial domain / periodicity
        self.L = 16 if L is None else L
        self.grid_size = (2 ** 4, 2 ** 6) if grid_size is None else grid_size
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device

        # Initialize WENO reconstrution object
        self.weno = WENO(self, order=3, device=self.device)
        self.fdm = FDM(self, device=self.device)
        self.force = None
        self.flux_splitting = f'godunov' if flux_splitting is None else flux_splitting

        assert (self.flux_splitting == f'godunov') or (self.flux_splitting == f'laxfriedrichs')


    def __repr__(self):
        return f'CE'

    def flux(self, input: torch.Tensor) -> torch.Tensor:
        """
        Flux as used in weno scheme for CE equations
        """
        return 0.5 * input ** 2

    def FDM_reconstruction(self, t: float, u):
        raise


    def WENO_reconstruction(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute derivatives using WENO scheme
        update = -alpha*uux + beta*uxx - gamma*uxxx
        weno reconstruction for uux
        FDM reconstruction gives uxx, uxxx
        Args:
            t (torch.Tensor): timepoint at which spatial terms are reconstructed, only important for time-dependent forcing term
            u (torch.Tensor): input fields at given timepoint
        Returns:
            torch.Tensor: reconstructed spatial derivatives
        """
        dudt = torch.zeros_like(u)

        # WENO reconstruction of advection term
        u_padded_weno = self.weno.pad(u)
        if self.flux_splitting == f'godunov':
            dudt = - self.alpha * self.weno.reconstruct_godunov(u_padded_weno, self.dx)

        if self.flux_splitting == f'laxfriedrichs':
            dudt = - self.alpha * self.weno.reconstruct_laxfriedrichs(u_padded_weno, self.dx)

        # reconstruction of diffusion term
        u_padded_fdm = self.fdm.pad(u)
        uxx = self.fdm.second_derivative(u_padded_fdm)
        uxxx = self.fdm.third_derivative(u_padded_fdm)

        dudt += self.beta*uxx
        dudt -= self.gamma*uxxx

        # Forcing term
        if self.force:
            dudt += self.force(t)

        return dudt


# class HDF5Dataset(Dataset):
#     """Load samples of an PDE Dataset, get items according to PDE"""

#     def __init__(self,
#                  path: str,
#                  pde: PDE,
#                  mode: str,
#                  base_resolution: list=None,
#                  super_resolution: list=None,
#                  load_all: bool=False,
#                  uniform_sample: int=-1,
#                  is_return_super = False,
#                 ) -> None:
#         """Initialize the dataset object
#         Args:
#             path: path to dataset
#             pde: string of PDE ('CE' or 'WE')
#             mode: [train, valid, test]
#             base_resolution: base resolution of the dataset [nt, nx]
#             super_resolution: super resolution of the dataset [nt, nx]
#             load_all: load all the data into memory
#         Returns:
#             None
#         """
#         super().__init__()
#         f = h5py.File(path, 'r')
#         self.mode = mode
#         self.pde = pde
#         self.dtype = torch.float64
#         self.data = f[self.mode]
#         self.base_resolution = (250, 100) if base_resolution is None else base_resolution
#         self.super_resolution = (250, 200) if super_resolution is None else super_resolution
#         self.uniform_sample = uniform_sample
#         self.dataset_base = f'pde_{self.base_resolution[0]}-{self.base_resolution[1]}'
#         self.dataset_super = f'pde_{self.super_resolution[0]}-{self.super_resolution[1]}'
#         self.is_return_super = is_return_super

#         if self.base_resolution[1] != 34:
#             ratio_nt = self.data[self.dataset_super].shape[1] / self.data[self.dataset_base].shape[1]
#             ratio_nx = self.data[self.dataset_super].shape[2] / self.data[self.dataset_base].shape[2]
#         else:
#             ratio_nt = 1.0
#             ratio_nx = int(200 / 33)
#         # assert (ratio_nt.is_integer())
#         # assert (ratio_nx.is_integer())
#         self.ratio_nt = int(ratio_nt)
#         self.ratio_nx = int(ratio_nx)

#         if self.base_resolution[1] != 34:
#             self.nt = self.data[self.dataset_base].attrs['nt']
#             self.dt = self.data[self.dataset_base].attrs['dt']
#             self.dx = self.data[self.dataset_base].attrs['dx']
#             self.x = self.data[self.dataset_base].attrs['x']
#             self.tmin = self.data[self.dataset_base].attrs['tmin']
#             self.tmax = self.data[self.dataset_base].attrs['tmax']
#         else:
#             self.nt = 250
#             self.dt = self.data['pde_250-50'].attrs['dt']
#             self.dx = 0.16 * 3
#             self.x = self.data['pde_250-100'].attrs['x'][::3]
#             self.tmin = self.data['pde_250-50'].attrs['tmin']
#             self.tmax = self.data['pde_250-50'].attrs['tmax']
#         self.x_ori = self.data['pde_250-100'].attrs['x']

#         if load_all:
#             data = {self.dataset_super: self.data[self.dataset_super][:]}
#             f.close()
#             self.data = data


#     def __len__(self):
#         #print(self.dataset_super)
#         return self.data[self.dataset_super].shape[0]

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
#         """
#         Get data item
#         Args:
#             idx (int): data index
#         Returns:
#             torch.Tensor: numerical baseline trajectory
#             torch.Tensor: downprojected high-resolution trajectory (used for training)
#             torch.Tensor: spatial coordinates
#             list: equation specific parameters
#         """
#         if(f'{self.pde}' == 'CE'):
#             # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
#             u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
#             left = u_super[..., -3:-1]
#             right = u_super[..., 1:3]
#             u_super_padded = torch.DoubleTensor(np.concatenate((left, u_super, right), -1))
#             weights = torch.DoubleTensor([[[[0.2]*5]]])
#             if self.uniform_sample == -1:
#                 u_super = F.conv1d(u_super_padded, weights, stride=(1, self.ratio_nx)).squeeze().numpy()
#             else:
#                 u_super = F.conv1d(u_super_padded, weights, stride=(1, 2)).squeeze().numpy()
#             x = self.x

#             # pdb.set_trace()
#             # Base resolution trajectories (numerical baseline) and equation specific parameters
#             if self.uniform_sample != -1:
#                 u_base = self.data[f'pde_{self.base_resolution[0]}-{100}'][idx][...,::self.uniform_sample]
#                 u_super_core = u_super[...,::self.uniform_sample]
#             else:
#                 u_base = self.data[self.dataset_base][idx]
#                 u_super_core = u_super
#             variables = {}
#             variables['alpha'] = self.data['alpha'][idx]
#             variables['beta'] = self.data['beta'][idx]
#             variables['gamma'] = self.data['gamma'][idx]

#             if self.is_return_super:
#                 return u_base, u_super_core, u_super, x, self.x_ori, variables
#             else:
#                 return u_base, u_super_core, x, variables

#         elif(f'{self.pde}' == 'WE'):
#             # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
#             # No padding is possible due to non-periodic boundary conditions
#             weights = torch.tensor([[[[1./self.ratio_nx]*self.ratio_nx]]])
#             u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
#             u_super = F.conv1d(torch.tensor(u_super), weights, stride=(1, self.ratio_nx)).squeeze().numpy()

#             # To match the downprojected trajectories, also coordinates need to be downprojected
#             x_super = torch.tensor(self.data[self.dataset_super].attrs['x'][None, None, None, :])
#             x = F.conv1d(x_super, weights, stride=(1, self.ratio_nx)).squeeze().numpy()

#             # Base resolution trajectories (numerical baseline) and equation specific parameters
#             u_base = self.data[self.dataset_base][idx]
#             variables = {}
#             variables['bc_left'] = self.data['bc_left'][idx]
#             variables['bc_right'] = self.data['bc_right'][idx]
#             variables['c'] = self.data['c'][idx]

#             return u_base, u_super, x, variables

#         else:
#             raise Exception("Wrong experiment")





class HDF5Dataset(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: PDE,
                 mode: str,
                 base_resolution: list=None,
                 super_resolution: list=None,
                 load_all: bool=False,
                 uniform_sample: int=-1,
                 is_return_super = False,
                ) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE ('CE' or 'WE')
            mode: [train, valid, test]
            base_resolution: base resolution of the dataset [nt, nx]
            super_resolution: super resolution of the dataset [nt, nx]
            load_all: load all the data into memory
        Returns:
            None
        """
        super().__init__()
        #import pdb; pdb.set_trace()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        self.dtype = torch.float64
        self.data = f[self.mode]
        self.base_resolution = (250, 100) if base_resolution is None else base_resolution
        self.super_resolution = (250, 200) if super_resolution is None else super_resolution
        self.uniform_sample = uniform_sample
        self.dataset_base = f'pde_{self.base_resolution[0]}-{self.base_resolution[1]}'
        self.dataset_super = f'pde_{self.super_resolution[0]}-{self.super_resolution[1]}'
        self.is_return_super = is_return_super

        if self.base_resolution[1] != 34:
            ratio_nt = self.data[self.dataset_super].shape[1] / self.data[self.dataset_base].shape[1]
            ratio_nx = self.data[self.dataset_super].shape[2] / self.data[self.dataset_base].shape[2]
        else:
            ratio_nt = 1.0
            ratio_nx = int(200 / 33)
        # assert (ratio_nt.is_integer())
        # assert (ratio_nx.is_integer())
        self.ratio_nt = int(ratio_nt)
        self.ratio_nx = int(ratio_nx)

        if self.base_resolution[1] != 34:
            self.nt = self.data[self.dataset_base].attrs['nt']
            self.dt = self.data[self.dataset_base].attrs['dt']
            self.dx = self.data[self.dataset_base].attrs['dx']
            self.x = self.data[self.dataset_base].attrs['x']
            self.tmin = self.data[self.dataset_base].attrs['tmin']
            self.tmax = self.data[self.dataset_base].attrs['tmax']
        else:
            self.nt = 250
            self.dt = self.data['pde_250-50'].attrs['dt']
            self.dx = 0.16 * 3
            self.x = self.data['pde_250-100'].attrs['x'][::3]
            self.tmin = self.data['pde_250-50'].attrs['tmin']
            self.tmax = self.data['pde_250-50'].attrs['tmax']
        self.x_ori = self.data['pde_250-100'].attrs['x']

        #import pdb; pdb.set_trace()
        if load_all:
            data = {self.dataset_super: self.data[self.dataset_super][:]}
            f.close()
            self.data = data


    def __len__(self):
        return self.data[self.dataset_super].shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: downprojected high-resolution trajectory (used for training)
            torch.Tensor: spatial coordinates
            list: equation specific parameters
        """
        if(f'{self.pde}' == 'CE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            #import pdb; pdb.set_trace()
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            left = u_super[..., -3:-1]
            right = u_super[..., 1:3]
            u_super_padded = torch.DoubleTensor(np.concatenate((left, u_super, right), -1))
            weights = torch.DoubleTensor([[[[0.2]*5]]])

            #import pdb; pdb.set_trace()
            if self.uniform_sample == -1:
                u_super = F.conv1d(u_super_padded, weights, stride=(1, self.ratio_nx)).squeeze().numpy()
            else:
                u_super = F.conv1d(u_super_padded, weights, stride=(1, 2)).squeeze().numpy()
            x = self.x

            # pdb.set_trace()
            # Base resolution trajectories (numerical baseline) and equation specific parameters
            if self.uniform_sample != -1:
                u_base = self.data[f'pde_{self.base_resolution[0]}-{100}'][idx][...,::self.uniform_sample]
                u_super_core = u_super[...,::self.uniform_sample]
            else:
                u_base = self.data[self.dataset_base][idx]
                u_super_core = u_super
            variables = {}
            variables['alpha'] = self.data['alpha'][idx]
            variables['beta'] = self.data['beta'][idx]
            variables['gamma'] = self.data['gamma'][idx]

            if self.is_return_super:
                return u_base, u_super_core, u_super, x, self.x_ori, variables
            else:
                return u_base, u_super_core, x, variables

        elif(f'{self.pde}' == 'WE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            # No padding is possible due to non-periodic boundary conditions
            weights = torch.tensor([[[[1./self.ratio_nx]*self.ratio_nx]]])
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            u_super = F.conv1d(torch.tensor(u_super), weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # To match the downprojected trajectories, also coordinates need to be downprojected
            x_super = torch.tensor(self.data[self.dataset_super].attrs['x'][None, None, None, :])
            x = F.conv1d(x_super, weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['bc_left'] = self.data['bc_left'][idx]
            variables['bc_right'] = self.data['bc_right'][idx]
            variables['c'] = self.data['c'][idx]

            return u_base, u_super, x, variables

        else:
            raise Exception("Wrong experiment")





























# def load_dataset_E1(args):

#         hdf5_train_file = h5py.File(args.dataset_train_path, 'r')
#         hdf5_test_file = h5py.File(args.dataset_test_path, 'r')

#         #import pdb; pdb.set_trace()
#         train_loaded_data = hdf5_train_file['train']['pde_250-200'][:]

#         #import pdb; pdb.set_trace()
#         test_loaded_data = hdf5_test_file['valid']['pde_250-200'][:]

#         train_tensor =  train_loaded_data.squeeze()
#         train_data = torch.from_numpy(train_tensor).float()
#         train_data = train_data.permute(0,2,1)

#         test_tensor =  test_loaded_data.squeeze()
#         test_data = torch.from_numpy(test_tensor).float()
#         test_data = test_data.permute(0,2,1)

#         # final_time = 4.98

#         # timestamps = torch.linspace(0,final_time,train_data.shape[-1])

#         # timestamps = (timestamps - timestamps.min())/(timestamps.max() - timestamps.min())

#         timestamps = torch.arange(0,250)*0.004
#         timestamps = timestamps.to(device)


#         x_train = train_data[:args.n_train,...]
#         x_test = test_data[:args.n_test,...]

#         x_train_test = train_data[:args.n_test,...]

#         res = x_train.shape[1]
#         train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train), batch_size=args.batch_size_train, shuffle=True)
#         test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test), batch_size=args.batch_size_test, shuffle=False)
#         train_test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_test), batch_size=args.batch_size_test, shuffle=False)

#         data = {"train_loader":train_loader, "test_loader": test_loader, "train_test_loader":train_test_loader,  "timestamps":timestamps}
#         return data








