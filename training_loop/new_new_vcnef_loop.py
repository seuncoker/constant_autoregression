import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from joblib import Parallel, delayed
#import wandb
# Import PDEBench dataloader


#from utils.dataset import PDEBenchDataset

# Import VCNeF models
# from vcnef.vcnef_1d import VCNeFModel as VCNeF1DModel
# from vcnef.vcnef_2d import VCNeFModel as VCNeF2DModel
# from vcnef.vcnef_3d import VCNeFModel as VCNeF3DModel

# Import function for counting model trainable parameters
#from utils.utils import count_model_params


import sys, os
import re
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

import math as mt
import h5py
from constant_autoregression.util import Printer, initialize_weights_xavier_uniform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = Printer(n_digits=6)

from constant_autoregression.model.vcnef_1d import VCNeFModel



class PDEBenchDataset(Dataset):
    """
    Loads data in PDEBench format. Slightly adaped code from PDEBench.
    """

    def __init__(self, filenames,
                 initial_step=10,
                 saved_folder='',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 truncated_trajectory_length=-1,
                 if_test=False,
                 test_ratio=0.5,
                 num_samples_max=-1):
        """
        Represent dataset that consists of PDE with different parameters.

        :param filenames: filenames that contain the datasets
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional
        :param truncated_trajectory_length: cuts temporal subsampled trajectory yielding a trajectory of given length. -1 means that trajectory is not truncated
        :type truncated_trajectory_length: INT, optional

        """

        # Also accept single file name
        if type(filenames) == str:
            filenames = [filenames]

        self.data = np.array([])
        self.pde_parameter = np.array([])

        # Load data
        def load(filename, num_samples_max, test_ratio):
            root_path = os.path.abspath(saved_folder + filename)
            assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'

            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()

                if 'tensor' not in keys:
                    _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                    idx_cfd = _data.shape
                    if len(idx_cfd)==3:  # 1D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              3],
                                             dtype=np.float32)
                        #density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,2] = _data   # batch, x, t, ch

                        grid = np.array(f["x-coordinate"], dtype=np.float32)
                        grid = torch.tensor(grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                        print(data.shape)
                    if len(idx_cfd)==4:  # 2D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              4],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,3] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

                    if len(idx_cfd)==5:  # 3D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              idx_cfd[4]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              5],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,3] = _data   # batch, x, t, ch
                        # Vz
                        _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,4] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        z = np.array(f["z-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        z = torch.tensor(z, dtype=torch.float)
                        X, Y, Z = torch.meshgrid(x, y, z)
                        grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution, \
                                    ::reduced_resolution, \
                                    ::reduced_resolution]

                else:  # scalar equations
                    ## data dim = [t, x1, ..., xd, v]
                    _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                    if len(_data.shape) == 3:  # 1D
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data = _data[:, :, :, None]  # batch, x, t, ch

                        grid = np.array(f["x-coordinate"], dtype=np.float32)
                        grid = torch.tensor(grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    if len(_data.shape) == 4:  # 2D Darcy flow
                        # u: label
                        _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        data = np.concatenate([_data, data], axis=-1)
                        data = data[:, :, :, :, None]  # batch, x, y, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

            if num_samples_max > 0:
                num_samples_max = min(num_samples_max, data.shape[0])
            else:
                num_samples_max = data.shape[0]

            test_idx = int(num_samples_max * test_ratio)

            if if_test:
                data = data[:test_idx]
            else:
                data = data[test_idx:num_samples_max]

            # Get pde parameter from file name
            matches = re.findall(r"_[a-zA-Z]+([0-9].[0-9]+|1.e-?[0-9]+)", filename)
            pde_parameter_scalar = [float(match) for match in matches]
            pde_parameter = np.tile(pde_parameter_scalar, (data.shape[0], 1)).astype(np.float32)

            return data, pde_parameter, grid

        
        data, pde_parameter, grid = zip(*Parallel(n_jobs=len(filenames))(delayed(load)(filename, num_samples_max, test_ratio) for filename in filenames))
        self.data = np.vstack(data)
        self.pde_parameter = np.vstack(pde_parameter)
        self.grid = grid[0]

        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.data = torch.tensor(self.data)
        self.pde_parameter = torch.tensor(self.pde_parameter)

        # truncate trajectory
        if truncated_trajectory_length > 0:
            self.data = self.data[..., :truncated_trajectory_length, :]        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        `self.data` is already subsampled across time and space.
        `self.grid` is already subsampled
        """
        return self.data[idx, ..., :self.initial_step, :], self.data[idx], self.grid, self.pde_parameter[idx]
    




















def run_training():
    """
    This training loop is an adapted version of the PDEBench training loop.
    """

    base_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/A1/"
    file_names = ["1D_Advection_Sols_beta0.1_subsampled.hdf5"]
    num_channels = 1
    pde_param_dim = 1
    condition_on_pde_param = False
    t_train = 41
    initial_step = 1
    reduced_resolution = 4
    reduced_resolution_t = 5
    reduced_batch = 8

    num_workers = 1
    model_update = 1
    model_path = "VCNeF.pt"

    batch_size = 64
    epochs = 500
    learning_rate = 3.e-4
    random_seed = 3407

    scheduler_warmup_fraction = 0.2

    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Initialize W&B
    #wandb.init()

    # Initialize the dataset and dataloader
    train_data = PDEBenchDataset(file_names,
                                 reduced_resolution=reduced_resolution,
                                 reduced_resolution_t=reduced_resolution_t,
                                 reduced_batch=reduced_batch,
                                 initial_step=initial_step,
                                 saved_folder=base_path)
    val_data = PDEBenchDataset(file_names,
                               reduced_resolution=reduced_resolution,
                               reduced_resolution_t=reduced_resolution_t,
                               reduced_batch=reduced_batch,
                               initial_step=initial_step,
                               if_test=True,
                               saved_folder=base_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    print(f"shape of train_data, 0: {train_data[-1][0].shape}")
    print(f"shape of train_data, 1: {train_data[-1][1].shape}")
    print(f"shape of train_data, 2: {train_data[-1][2].shape}")
    print(f"shape of train_data, 3: {train_data[-1][3].shape}")
    print(f"shape of val_data, 0: {val_data[-1][0].shape}")
    print(f"shape of val_data, 1: {val_data[-1][1].shape}")
    print(f"shape of val_data, 2: {val_data[-1][2].shape}")
    print(f"shape of val_data, 3: {val_data[-1][3].shape}")
    print(f"length of train_loader: {len(train_loader)}")
    print(f"length of val_loader: {len(val_loader)}")

    _, _data, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print("Spatial Dimension", dimensions - 3)

    # Set up model
    if dimensions == 4:
        print("VCNeF 1D")
        model = VCNeFModel(d_model=96,
                            n_heads=8,
                            num_channels=num_channels,
                            condition_on_pde_param=condition_on_pde_param,
                            pde_param_dim=pde_param_dim,
                            n_transformer_blocks=3,
                            n_modulation_blocks=3).to(device)
    # elif dimensions == 5:
    #     print("VCNeF 2D")
    #     model = VCNeF2DModel(d_model=256,
    #                          n_heads=8,
    #                          num_channels=num_channels,
    #                          condition_on_pde_param=condition_on_pde_param,
    #                          pde_param_dim=pde_param_dim,
    #                          n_transformer_blocks=1,
    #                          n_modulation_blocks=6)
    # elif dimensions == 6:
    #     print("VCNeF 3D")
    #     model = VCNeF3DModel(d_model=256,
    #                          n_heads=8,
    #                          num_channels=num_channels,
    #                          condition_on_pde_param=condition_on_pde_param,
    #                          pde_param_dim=pde_param_dim,
    #                          n_transformer_blocks=1,
    #                          n_modulation_blocks=6)
    # model.to(device)
    # total_params = count_model_params(model)
    # print(f"Total Trainable Parameters = {total_params}")

    # Set maximum time step of the data to train
    if t_train > _data.shape[-2]:
        t_train = _data.shape[-2]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=learning_rate,
                                                    pct_start=scheduler_warmup_fraction,
                                                    div_factor=1e3,
                                                    final_div_factor=1e4,
                                                    total_steps=epochs * len(train_loader))

    loss_fn = nn.MSELoss(reduction="mean")
    loss_fn_no_reduction = nn.MSELoss(reduction="none")
    loss_val_min = np.inf

    for ep in range(epochs):
        model.train()
        train_l2_step = 0
        train_l2_full = 0
        train_l2_full_mean = 0

        for xx, yy, grid, pde_param in train_loader:
            #p.print(f"x_train, y_train, grid, pde_param: {xx.shape}, {yy.shape} {grid.shape} {pde_param.shape}")
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)
            pde_param = pde_param.to(device)

            yy_train = yy[..., 0:t_train, :]

            # Prepare queried times t in [0..1]
            t = torch.arange(initial_step, t_train, device=xx.device) * 1 / (t_train-1)
            t = t.repeat((xx.size(0), 1)).to(device)

            # Forward pass
            #p.print(f"xx, grid, pde_param, t: {xx[..., 0, :].shape} {grid.shape} {pde_param.shape} {t.shape}")
            pred_train = model(xx[..., 0, :], grid, pde_param, t)
            pred_train = torch.cat((xx, pred_train), dim=-2)

            # Loss calculation
            _batch = yy.size(0)
            loss = torch.sum(torch.mean(loss_fn_no_reduction(pred_train.unsqueeze(-1), yy_train.unsqueeze(-1)), dim=(0, 1)))
            l2_full = loss_fn(pred_train.reshape(_batch, -1), yy_train.reshape(_batch, -1)).item()
            train_l2_step += loss.item()
            train_l2_full += l2_full
            train_l2_full_mean += l2_full * _batch

            c_lr = optimizer.param_groups[0]['lr']

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            scheduler.step()

        if ep % model_update == 0:
            val_l2_step = 0
            val_l2_full = 0
            val_l2_full_mean = 0
            model.eval()

            with torch.no_grad():
                for xx, yy, grid, pde_param in val_loader:
                    loss = 0
                    xx = xx.to(device)
                    yy = yy.to(device)
                    grid = grid.to(device)
                    pde_param = pde_param.to(device)

                    # Prepare queried times t in [0..1]
                    t = torch.arange(initial_step, yy.shape[-2], device=xx.device) * 1 / (t_train-1)
                    t = t.repeat((xx.size(0), 1)).to(device)

                    # Forward pass
                    pred = model(xx[..., 0, :], grid, pde_param, t)
                    pred = torch.cat((xx, pred), dim=-2)

                    # Loss calculation
                    _batch = yy.size(0)
                    loss = torch.sum(torch.mean(loss_fn_no_reduction(pred.unsqueeze(-1), yy.unsqueeze(-1)), dim=(0, 1)))
                    l2_full = loss_fn(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()
                    val_l2_step += loss.item()
                    val_l2_full += l2_full
                    val_l2_full_mean += l2_full * _batch

                # Calculate mean of l2 full loss
                train_l2_full_mean = train_l2_full_mean / len(train_loader.dataset)
                val_l2_full_mean = val_l2_full_mean / len(val_loader.dataset)

                # Save checkpoint
                # if val_l2_full < loss_val_min:
                #     loss_val_min = val_l2_full
                #     torch.save({
                #         "epoch": ep,
                #         "model_state_dict": model.state_dict(),
                #         "optimizer_state_dict": optimizer.state_dict(),
                #         "scheduler_state_dict": scheduler.state_dict(),
                #         "loss": loss_val_min
                #     }, model_path)
            model.train()

        # Log metrics in W&B
        # wandb.log({
        #     "train/loss": train_l2_full,
        #     "train/mean_loss": train_l2_full_mean,
        #     "val/loss": val_l2_full,
        #     "val/mean_loss": val_l2_full_mean,
        #     "lr": scheduler.get_last_lr()[0]
        # })

        p.print(f" ep: {ep}, lr:{c_lr}, train_l2_full_mean: {train_l2_full_mean}, val_l2_full_mean:{val_l2_full_mean} ")
        p.print("\n")
        # })
    torch.save({"pred": pred, "actual": yy, "pred_train": pred_train, "actual_train": yy_train}, "vcnef_result_4.pt")

if __name__ == "__main__":
    run_training()
    print("Done.")