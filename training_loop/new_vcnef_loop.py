import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


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

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))


from constant_autoregression.util import Printer, initialize_weights_xavier_uniform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = Printer(n_digits=6)

from constant_autoregression.model.vcnef_1d import VCNeFModel


def load_dataset_A1():
        n_train = 64
        n_test= 32

        dataset_train_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/A1/1D_Advection_Sols_beta0.5_K1_N2_Sa2500.npy"
        dataset_valid_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/A1/1D_Advection_Sols_beta0.5_K1_N2_Sa2500.npy"
        dataset_test_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/A1/1D_Advection_Sols_beta0.5_K1_N2_Sa2500.npy"
        hdf5_train_file = np.load(dataset_train_path)
        hdf5_test_file = np.load(dataset_test_path)
        hdf5_valid_file = np.load(dataset_valid_path)

        train_loaded_data = hdf5_train_file[:2100]
        test_loaded_data =  hdf5_test_file[-128:]
        valid_loaded_data = hdf5_valid_file[-256:-128]

        #train_tensor =  train_loaded_data.squeeze()
        train_data = torch.from_numpy(train_loaded_data).float()
        #train_data = train_data - train_data.mean(-1).unsqueeze(-1)
        train_data = train_data[:n_train,::5,::4]

        #test_tensor =  test_loaded_data.squeeze()
        test_data = torch.from_numpy(test_loaded_data).float()
        #test_data = test_data - test_data.mean(-1).unsqueeze(-1)
        test_data = test_data[:n_test,::5,::4]

        #valid_tensor =  valid_loaded_data.squeeze()
        valid_data = torch.from_numpy(valid_loaded_data).float()
        #valid_data = valid_data - valid_data.mean(-1).unsqueeze(-1)
        valid_data = valid_data[:n_test,::5,::4]
        
        x_train = train_data[:,:-1,:].permute(0,2,1).unsqueeze(-1)
        x_test = test_data[:,:-1,:].permute(0,2,1).unsqueeze(-1)
        x_valid = valid_data[:,:-1,:].permute(0,2,1).unsqueeze(-1)

        y_train = train_data[:,1:,:].permute(0,2,1).unsqueeze(-1)
        y_test = test_data[:,1:,:].permute(0,2,1).unsqueeze(-1)
        y_valid = valid_data[:,1:,:].permute(0,2,1).unsqueeze(-1)

        p.print(f"x_train, y_train: {x_train.shape}, {y_train.shape}")
        #import pdb; pdb.set_trace()
        #args.t_resolution =  x_train.shape[2]
        # x_resolution =  x_train.shape[1]

        # if t_resolution_train == None:
        #         t_resolution_train = x_train.shape[2]

        # if t_resolution_test == None:
        #         t_resolution_test =  x_test.shape[2]

        # if t_resolution_valid == None:
        #         t_resolution_valid = x_valid.shape[2]
                

        # #args.timestamps = [i for i in range(args.t_resolution)]
        # args.timestamps_valid = [i*0.01 for i in range(args.t_resolution_valid)]
        # args.timestamps_test = [i*0.01 for i in range(args.t_resolution_test)]
        # args.timestamps_train = [i*0.01 for i in range(args.t_resolution_train)]

        size_x =x_train.shape[1]
        batch_size = x_train.shape[0]
        batch_size_ = x_test.shape[0]

        grid = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).reshape(1, size_x, 1, 1).repeat([batch_size, 1, 1, 1]).to(device)
        grid_ = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).reshape(1, size_x, 1, 1).repeat([batch_size_, 1, 1, 1]).to(device)
        
        p.print(f"x_train, y_train, grid : {x_train.shape}, {y_train.shape} {grid.shape}")
        p.print(f"x_test, y_test, grid : {x_test.shape}, {y_test.shape} {grid.shape}")
        p.print(f"x_valid, y_valid, grid : {x_valid.shape}, {y_valid.shape} {grid.shape}")

        #res = x_train.shape[1]
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,y_train, grid, x_train), batch_size=n_train, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, grid_, x_test), batch_size=n_test, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, y_valid, grid_, x_valid), batch_size=n_test, shuffle=False)


        # train_loader = torch.utils.data.DataLoader((x_train,y_train, grid, x_train), batch_size=n_train, shuffle=True)
        # test_loader = torch.utils.data.DataLoader((x_test, y_test, grid, x_test), batch_size=n_test, shuffle=False)
        # valid_loader = torch.utils.data.DataLoader((x_valid, y_valid, grid, x_valid), batch_size=n_test, shuffle=False)

        #data = {"train_loader":train_loader, "test_loader": test_loader, "timestamps":timestamps}
        #import pdb; pdb.set_trace()
        #p.print(f"timestamps_test: {args.timestamps_test[:10]}" )
        return train_loader,valid_loader, test_loader



def run_training():
    """
    This training loop is an adapted version of the PDEBench training loop.
    """

    base_path = "pdebench/data/"
    file_names = ["1D_Burgers_Sols_Nu0.001.hdf5"]
    num_channels = 1
    pde_param_dim = 1
    condition_on_pde_param = False
    t_train = 41
    initial_step = 1
    reduced_resolution = 4
    reduced_resolution_t = 5
    reduced_batch = 1

    num_workers = 8
    model_update = 1
    model_path = "VCNeF.pt"

    batch_size = 32
    epochs = 500
    learning_rate = 1.e-3
    random_seed = 3407

    scheduler_warmup_fraction = 0.2

    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Initialize W&B
    # wandb.init()

    # # Initialize the dataset and dataloader
    # train_data = PDEBenchDataset(file_names,
    #                              reduced_resolution=reduced_resolution,
    #                              reduced_resolution_t=reduced_resolution_t,
    #                              reduced_batch=reduced_batch,
    #                              initial_step=initial_step,
    #                              saved_folder=base_path)
    # val_data = PDEBenchDataset(file_names,
    #                            reduced_resolution=reduced_resolution,
    #                            reduced_resolution_t=reduced_resolution_t,
    #                            reduced_batch=reduced_batch,
    #                            initial_step=initial_step,
    #                            if_test=True,
    #                            saved_folder=base_path)
    # train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)


    train_loader,valid_loader, test_loader =  load_dataset_A1()

    # print(f"shape of train_data, 0: {train_data[-1][0].shape}")
    # print(f"shape of train_data, 1: {train_data[-1][1].shape}")
    # print(f"shape of train_data, 2: {train_data[-1][2].shape}")
    # print(f"shape of train_data, 3: {train_data[-1][3].shape}")
    # print(f"shape of val_data, 0: {val_data[-1][0].shape}")
    # print(f"shape of val_data, 1: {val_data[-1][1].shape}")
    # print(f"shape of val_data, 2: {val_data[-1][2].shape}")
    # print(f"shape of val_data, 3: {val_data[-1][3].shape}")
    # print(f"length of train_loader: {len(train_loader)}")
    # print(f"length of val_loader: {len(val_loader)}")

    # _, _data, _, _ = next(iter(val_loader))
    # dimensions = len(_data.shape)
    # print("Spatial Dimension", dimensions - 3)

    # Set up model



    model = VCNeFModel(d_model=96,
                        n_heads=8,
                        num_channels=num_channels,
                        condition_on_pde_param=condition_on_pde_param,
                        pde_param_dim=pde_param_dim,
                        n_transformer_blocks=3,
                        n_modulation_blocks=3).to(device)
    
    # if dimensions == 4:
    #     print("VCNeF 1D")
    #     model = VCNeF1DModel(d_model=96,
    #                         n_heads=8,
    #                         num_channels=num_channels,
    #                         condition_on_pde_param=condition_on_pde_param,
    #                         pde_param_dim=pde_param_dim,
    #                         n_transformer_blocks=3,
    #                         n_modulation_blocks=3)
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
    # if t_train > _data.shape[-2]:
    #     t_train = _data.shape[-2]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
    #                                                 max_lr=learning_rate,
    #                                                 pct_start=scheduler_warmup_fraction,
    #                                                 div_factor=1e3,
    #                                                 final_div_factor=1e4,
    #                                                 total_steps=epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size= 20,
                                                gamma=0.9)
    loss_fn = nn.MSELoss(reduction="mean")
    loss_fn_no_reduction = nn.MSELoss(reduction="none")
    loss_val_min = np.inf

    for ep in range(epochs):
        model.train()
        train_l2_step = 0
        train_l2_full = 0
        train_l2_full_mean = 0

        for (xx, yy, grid, pde_param) in train_loader:
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
            # p.print("Forward pass")
            # p.print(f"xx, grid, pde_param, t: {xx[..., 0, :].shape} {grid[..., 0, :].shape} {pde_param.shape} {t.shape}")
            pred_train = model(xx[..., 0, :], grid[..., 0, :], pde_param, t)
            #pred = torch.cat((xx, pred), dim=-2)

            # Loss calculation
            _batch = yy_train.size(0)
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
                for (xx, yy, grid, pde_param) in valid_loader:
                    loss = 0
                    xx = xx.to(device)
                    yy = yy.to(device)
                    grid = grid.to(device)
                    pde_param = pde_param.to(device)

                    # Prepare queried times t in [0..1]
                    t = torch.arange(initial_step, yy.shape[-2]+1, device=xx.device) * 1 / (t_train-1)
                    t = t.repeat((xx.size(0), 1)).to(device)

                    # Forward pass
                    pred = model(xx[..., 0, :], grid[..., 0, :], pde_param, t)
                    #pred = torch.cat((xx, pred), dim=-2)

                    # Loss calculation
                    _batch = yy.size(0)
                    #p.print(f"pred, yy: {pred.shape}, {yy.shape}")
                    loss = torch.sum(torch.mean(loss_fn_no_reduction(pred.unsqueeze(-1), yy.unsqueeze(-1)), dim=(0, 1)))
                    l2_full = loss_fn(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()
                    val_l2_step += loss.item()
                    val_l2_full += l2_full
                    val_l2_full_mean += l2_full * _batch

                # Calculate mean of l2 full loss
                train_l2_full_mean = train_l2_full_mean / len(train_loader.dataset)
                val_l2_full_mean = val_l2_full_mean / len(valid_loader.dataset)

                # Save checkpoint
                if val_l2_full < loss_val_min:
                    loss_val_min = val_l2_full
                    # torch.save({
                    #     "epoch": ep,
                    #     "model_state_dict": model.state_dict(),
                    #     "optimizer_state_dict": optimizer.state_dict(),
                    #     "scheduler_state_dict": scheduler.state_dict(),
                    #     "loss": loss_val_min
                    # }, model_path)
            model.train()
        p.print(f" ep: {ep}, lr:{c_lr}, train_l2_full_mean: {train_l2_full_mean}, val_l2_full_mean:{val_l2_full_mean} ")
        p.print("\n")

        # Log metrics in W&B
        # wandb.log({
        #     "train/loss": train_l2_full,
        #     "train/mean_loss": train_l2_full_mean,
        #     "val/loss": val_l2_full,
        #     "val/mean_loss": val_l2_full_mean,
        #     "lr": scheduler.get_last_lr()[0]
        # })
    torch.save({"pred": pred, "actual": yy, "pred_train": pred_train, "actual_train": yy_train}, "vcnef_result_2.pt")

if __name__ == "__main__":
    run_training()
    p.print("Done.")