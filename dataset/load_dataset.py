import pdb
import pickle
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import h5py
import numpy as np
from variable_autoregression.util import Printer, create_data, Normalizer_1D
from variable_autoregression.dataset.mppde1d import CE, HDF5Dataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p = Printer(n_digits=6)
normalizer = Normalizer_1D()


def load_data(args, **kwargs):

    # train_loader = None
    # valid_loader = None
    # test_loader = None



    if args.dataset_name.endswith("E1"):
        args.total_t_range = 250
        args.x_res = 100
        uniform_sample = -1
        super_resolution = [args.total_t_range,200]
        base_resolution = [args.total_t_range,100]
        n_workers = 4
        pde = CE(device=device)

        args.time_stamps = [i*0.004 for i in range(0,args.total_t_range)]

        train_string = f'dataset/data/{pde}_train_E1.h5'
        valid_string = f'dataset/data/{pde}_valid_E1.h5'
        test_string = f'dataset/data/{pde}_test_E1.h5'

        p.print(f"Load dataset: {train_string}")
        train_dataset = HDF5Dataset(train_string, pde=pde, mode='train', base_resolution=base_resolution, super_resolution=super_resolution, uniform_sample=uniform_sample)
        #import pdb; pdb.set_trace()
        #train_dataset[f'pde_{super_resolution[0]}-{super_resolution[1]}'][:args.n_train]
        
        p.print(f"Load dataset: {valid_string}")
        valid_dataset = HDF5Dataset(valid_string, pde=pde, mode='valid', base_resolution=base_resolution, super_resolution=super_resolution, uniform_sample=uniform_sample)
        #valid_dataset[f'pde_{super_resolution[0]}-{super_resolution[1]}'][:args.n_test]

        p.print(f"Load dataset: {test_string}")
        test_dataset = HDF5Dataset(test_string, pde=pde, mode='test', base_resolution=base_resolution, super_resolution=super_resolution, uniform_sample=uniform_sample)
        #test_dataset[f'pde_{super_resolution[0]}-{super_resolution[1]}'][:args.n_test]

    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size_train,
                                shuffle=True,
                                num_workers= n_workers,
                                )

    valid_loader = DataLoader(valid_dataset,
                            batch_size=args.batch_size_test,
                            shuffle=False,
                            num_workers=n_workers,
                            )

    test_loader = DataLoader(test_dataset,
                                batch_size=args.batch_size_test,
                                shuffle=False,
                                num_workers=n_workers,
                                )
    
    #import pdb; pdb.set_trace()
    return train_loader, valid_loader, test_loader

    








# class data_random_batch():
#     """Object for holding a batch of data with mask during training."""

#     def __init__(self, input_sample_type, total_range, no_of_samp, dt_input, t_pred_steps, horizon):
#         indicies_1 = self.input_indicies_1(total_range, (no_of_samp[0]//4,no_of_samp[1]) )
#         indicies_2 = self.input_indicies_2(total_range, (no_of_samp[0]//4,no_of_samp[1]), dt_input)
#         indicies_3 = self.input_indicies_3(total_range, (no_of_samp[0]//4,no_of_samp[1]), dt_input, t_pred_steps, horizon)
#         indicies_4 = self.input_indicies_4(total_range, (no_of_samp[0]//4,no_of_samp[1]), dt_input, t_pred_steps, horizon  )

#         #import pdb; pdb.set_trace()

#         self.indicies = torch.cat((indicies_1, indicies_2, indicies_3, indicies_4 ), dim=0).long()


#     @staticmethod
#     def input_indicies_1(total_range, no_of_samp):
#         """
#         generate n random input samples from the range (250 )
#         """
#         indicies = torch.sort(torch.randint(total_range, (no_of_samp)))[0]
#         return indicies

#     @staticmethod
#     def input_indicies_2(total_range, no_of_samp, dt_input ):
#         """
#         generate no_of_samp constant dt input independent samples from the range (0, input_range )
#         """

#         #import pdb; pdb.set_trace()
#         dts = torch.randint(1, dt_input + 1, size=(no_of_samp[0],))
#         indicies = torch.ones((no_of_samp) )
#         for i in range(no_of_samp[0]):
#             indicies[i,:] = torch.arange(0,total_range)[::dts[i]][:no_of_samp[1]]
        
#         #import pdb; pdb.set_trace()
#         return indicies

#     @staticmethod
#     def input_indicies_3(total_range, no_of_samp, dt_input, t_pred_steps, horizon  ):
#         """
#         generate no_of_samp such that the initial time is randomly generated and 
#         """

#         #import pdb; pdb.set_trace()
#         init_time_stamp_range = torch.tensor([t for t in range(0, total_range -  no_of_samp[1])])
#         random_steps = init_time_stamp_range[torch.randint(len(init_time_stamp_range), (no_of_samp[0],))]
#         indicies = torch.ones((no_of_samp) )
#         for i in range(no_of_samp[0]):
#             indicies[i] = torch.arange(random_steps[i],random_steps[i]+ no_of_samp[1])

#         #import pdb; pdb.set_trace()
#         return indicies

#     @staticmethod
#     def input_indicies_4(total_range, no_of_samp, dt_input, t_pred_steps, horizon  ):
#         """
#         generate no_of_samp such that the initial time is randomly generated and 
#         """

#         #import pdb; pdb.set_trace()
#         dts = torch.randint(2, dt_input + 1, size=(no_of_samp[0],))
#         init_time_stamp_range = torch.tensor([t for t in range(0, total_range -  no_of_samp[1]*2)])
#         random_steps = init_time_stamp_range[torch.randint(len(init_time_stamp_range), (no_of_samp[0],))]
#         indicies = torch.ones((no_of_samp) )
#         for i in range(no_of_samp[0]):
#             indicies[i] = torch.arange(random_steps[i],total_range, 2)[:no_of_samp[1]]

#         #import pdb; pdb.set_trace()
#         return indicies





# class data_random_batch():
#     """Object for holding a batch of data with mask during training."""

#     def __init__(self, input_sample_type, total_range, no_of_samp, dt_input, t_pred_steps, horizon):
#         if input_sample_type == 1:
#             self.indicies = self.input_indicies_1(total_range, no_of_samp)
#         elif input_sample_type == 2:
#             self.indicies = self.input_indicies_2(total_range, no_of_samp, dt_input)
#         elif input_sample_type == 3:
#             self.indicies = self.input_indicies_3(total_range, no_of_samp, dt_input, t_pred_steps, horizon)
#         else:
#             raise TypeError("Specify input_sample_type: 1 (non_independent sampes ) OR 2 (Independent samples )")
#         #self.input = data[..., self.input_indicies]

#     @staticmethod
#     def input_indicies_1(total_range, no_of_samp):
#         """
#         generate n random input samples from the range (250 )
#         """
#         indicies = torch.sort(torch.randint(total_range, (no_of_samp)))[0]
#         return indicies

#     @staticmethod
#     def input_indicies_2(total_range, no_of_samp, dt_input ):
#         """
#         generate no_of_samp constant dt input independent samples from the range (0, input_range )
#         """

#         import pdb; pdb.set_trace()
#         dts = torch.randint(1, dt_input + 1, size=(no_of_samp[0],))
#         indicies = torch.ones((no_of_samp) )
#         for i in range(no_of_samp[0]):
#             indicies[i,:] = torch.arange(0,total_range)[::dts[i]][:no_of_samp[1]]
        
#         import pdb; pdb.set_trace()
#         return indicies

#     @staticmethod
#     def input_indicies_3(total_range, no_of_samp, dt_input,  ):
#         """
#         generate no_of_samp such that the initial time is randomly generated and 
#         """

#         import pdb; pdb.set_trace()
#         init_time_stamp_range = torch.tensor([t for t in range(0, total_range -  no_of_samp[1])])
#         random_steps = init_time_stamp_range[torch.randint(len(init_time_stamp_range), (no_of_samp[0],))]
#         indicies = torch.ones((no_of_samp) )
#         for i in range(no_of_samp[0]):
#             indicies[i] = torch.arange(random_steps,random_steps+ no_of_samp[1])

#         import pdb; pdb.set_trace()
#         return indicies
    



# class Input_Batch():
#     """Object for holding a batch of data with mask during training."""

#     def __init__(self, input_sample_type, input_range, total_range, no_of_input, dt_input = None ):
#         if input_sample_type == 1:
#             self.input_indicies = self.input_indicies_1(input_range, no_of_input)
#         elif input_sample_type == 2:
#             self.input_indicies = self.input_indicies_2(input_range, no_of_input, dt_input)
#         elif input_sample_type == 3:
#             assert dt_input != None
#             self.input_indicies = self.input_indicies_3(input_range, no_of_input, dt_input)
#         else:
#             raise TypeError("Specify input_sample_type: 1 (non_independent sampes ) OR 2 (Independent samples )")
#         #self.input = data[..., self.input_indicies]

#     @staticmethod
#     def input_indicies_1(input_range, n):
#         """
#         generate n random input samples from the range (0, input_range )
#         """
#         return torch.sort(torch.randint(input_range, (n,)))[0]

#     @staticmethod
#     def input_indicies_2(input_range, n, dt_input ):
#         """
#         generate n constant dt input independent samples from the range (0, input_range )
#         """
#         assert dt_input < 6
#         return torch.arange(0,input_range,1)[::dt_input][:n]




# class Output_Batch():
#     """Object for holding a batch of data with mask during training."""

#     def __init__(self,input_indicies=None, output_sample_type=None, total_range=None, no_of_output=None ):
#         if output_sample_type == 1:
#             self.output_indicies = self.output_indicies_1(input_indicies, no_of_output, total_range)
#         elif output_sample_type == 2:
#             self.output_indicies = self.output_indicies_2(input_indicies, no_of_output, total_range)
#         else:
#             raise TypeError("Specify input_sample_type: 1 (non_independent sampes ) OR 2 (Independent samples )")
#         #self.output = data[..., self.output_indicies]

#     @staticmethod
#     def output_indicies_1(input_indicies, no_of_output,total_range):
#         """
#         generate n variable dt output samples from the remining indicies
#         """
#         output_indicies = torch.sort(torch.randint(total_range-input_indicies[-1]-1, (no_of_output,)))[0]
#         return output_indicies + input_indicies[-1]+ 1


#     @staticmethod
#     def output_indicies_2(input_indicies, no_of_output,total_range):
#         """
#         generate n constant dt output samples from the remining indicies
#         """
#         last_ind_id = input_indicies[-1]
#         dt = int(1)
#         return torch.arange(last_ind_id+dt, total_range)[::dt] #output_indicies  #+ input_indicies[-1] + 1





# class no_of_output_space():
#     """Generate a space of number of output predictions

#     args: out_low = smallest number of output predictions
#           out_low = highest number of output predictions

#     return:
#       output_space: space of number of output predictions
#       output_tray: number of samples from output space
#     """

#     def __init__(self, out_high=None, output_space_type=1, t_pred_steps=None, horizon=None, total_range=None, step = None, predefined=None):
#         if output_space_type == 1:
#             self.output_tray = self.output_space_1(horizon, t_pred_steps, total_range, step)

#         elif output_space_type == 3:
#             self.output_tray = predefined
#         else:
#             raise TypeError("Specify output_space_type: 1 ")


#     @staticmethod
#     def output_space_1(horizon, t_pred_steps, total_range, n = 25):
#         """
#         generate n output space
#         """
#         return torch.arange(t_pred_steps*(horizon+1),total_range+1, n )


def load_dataset_A1(args):

        hdf5_train_file = np.load(args.dataset_train_path)
        hdf5_test_file = np.load(args.dataset_test_path)
        hdf5_valid_file = np.load(args.dataset_valid_path)

        train_loaded_data = hdf5_train_file[:2100]
        test_loaded_data =  hdf5_test_file[-128:]
        valid_loaded_data = hdf5_valid_file[-256:-128]

        train_tensor =  train_loaded_data.squeeze()
        train_data = torch.from_numpy(train_tensor).float()
        #train_data = train_data - train_data.mean(-1).unsqueeze(-1)
        train_data = train_data[...,::4]

        test_tensor =  test_loaded_data.squeeze()
        test_data = torch.from_numpy(test_tensor).float()
        #test_data = test_data - test_data.mean(-1).unsqueeze(-1)
        test_data = test_data[...,::4]

        valid_tensor =  valid_loaded_data.squeeze()
        valid_data = torch.from_numpy(valid_tensor).float()
        #valid_data = valid_data - valid_data.mean(-1).unsqueeze(-1)
        valid_data = valid_data[...,::4]
        
        if args.subanalysis_type == "one_sample":
               x_train = train_data[4:4+args.n_train,...].permute(0,2,1)  ####  CHANGE HERE
        else:
               x_train = train_data[:args.n_train,...].permute(0,2,1)


        x_test = test_data[:args.n_test,...].permute(0,2,1)
        x_valid = valid_data[:args.n_test,...].permute(0,2,1)
        
        #import pdb; pdb.set_trace()
        #args.t_resolution =  x_train.shape[2]
        args.x_resolution =  x_train.shape[1]

        if args.t_resolution_train == None:
                args.t_resolution_train = x_train.shape[2]

        if args.t_resolution_test == None:
                args.t_resolution_test =  x_test.shape[2]

        if args.t_resolution_valid == None:
                args.t_resolution_valid = x_valid.shape[2]
                

        #args.timestamps = [i for i in range(args.t_resolution)]
        args.timestamps_valid = [i*0.01 for i in range(args.t_resolution_valid)]
        args.timestamps_test = [i*0.01 for i in range(args.t_resolution_test)]
        args.timestamps_train = [i*0.01 for i in range(args.t_resolution_train)]


        res = x_train.shape[1]
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,x_train, x_train, x_train), batch_size=args.batch_size_train, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, x_test, x_test), batch_size=args.batch_size_test, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, x_valid, x_valid, x_valid), batch_size=args.batch_size_test, shuffle=False)

        #data = {"train_loader":train_loader, "test_loader": test_loader, "timestamps":timestamps}
        #import pdb; pdb.set_trace()
        #p.print(f"timestamps_test: {args.timestamps_test[:10]}" )
        return train_loader,valid_loader, test_loader




def load_dataset_B1(args):


        hdf5_train_file = np.load(args.dataset_train_path)
        hdf5_test_file = np.load(args.dataset_test_path)
        hdf5_valid_file = np.load(args.dataset_valid_path)

        train_loaded_data = hdf5_train_file[0, :2100]
        test_loaded_data =  hdf5_test_file[0, -128:]
        valid_loaded_data = hdf5_valid_file[0, -256:-128]

        train_tensor =  train_loaded_data.squeeze()
        train_data = torch.from_numpy(train_tensor).float()
        #train_data = train_data - train_data.mean(-1).unsqueeze(-1)
        train_data = train_data[...,::4]

        test_tensor =  test_loaded_data.squeeze()
        test_data = torch.from_numpy(test_tensor).float()
        #test_data = test_data - test_data.mean(-1).unsqueeze(-1)
        test_data = test_data[...,::4]

        valid_tensor =  valid_loaded_data.squeeze()
        valid_data = torch.from_numpy(valid_tensor).float()
        #valid_data = valid_data - valid_data.mean(-1).unsqueeze(-1)
        valid_data = valid_data[...,::4]
        
        if args.subanalysis_type == "one_sample":
               x_train = train_data[4:4+args.n_train,...].permute(0,2,1)  ####  CHANGE HERE
        else:
               x_train = train_data[:args.n_train,...].permute(0,2,1)


        x_test = test_data[:args.n_test,...].permute(0,2,1)
        x_valid = valid_data[:args.n_test,...].permute(0,2,1)
        
        #import pdb; pdb.set_trace()
        #args.t_resolution =  x_train.shape[2]
        args.x_resolution =  x_train.shape[1]

        if args.t_resolution_train == None:
                args.t_resolution_train = x_train.shape[2]

        if args.t_resolution_test == None:
                args.t_resolution_test =  x_test.shape[2]

        if args.t_resolution_valid == None:
                args.t_resolution_valid = x_valid.shape[2]
                

        #args.timestamps = [i for i in range(args.t_resolution)]
        args.timestamps_valid = [i*0.01 for i in range(args.t_resolution_valid)]
        args.timestamps_test = [i*0.01 for i in range(args.t_resolution_test)]
        args.timestamps_train = [i*0.01 for i in range(args.t_resolution_train)]


        res = x_train.shape[1]
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,x_train, x_train, x_train), batch_size=args.batch_size_train, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, x_test, x_test), batch_size=args.batch_size_test, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, x_valid, x_valid, x_valid), batch_size=args.batch_size_test, shuffle=False)

        #data = {"train_loader":train_loader, "test_loader": test_loader, "timestamps":timestamps}
        #import pdb; pdb.set_trace()
        #p.print(f"timestamps_test: {args.timestamps_test[:10]}" )
        return train_loader,valid_loader, test_loader




def load_dataset_E1(args):

        hdf5_train_file = h5py.File(args.dataset_train_path, 'r')
        hdf5_test_file = h5py.File(args.dataset_test_path, 'r')
        hdf5_valid_file = h5py.File(args.dataset_valid_path, 'r')

        train_loaded_data = hdf5_train_file['train']['pde_250-200'][:]
        test_loaded_data = hdf5_test_file['test']['pde_250-200'][:]
        valid_loaded_data = hdf5_valid_file['valid']['pde_250-200'][:]

        train_tensor =  train_loaded_data.squeeze()
        train_data = torch.from_numpy(train_tensor).float()

        test_tensor =  test_loaded_data.squeeze()
        test_data = torch.from_numpy(test_tensor).float()

        valid_tensor =  valid_loaded_data.squeeze()
        valid_data = torch.from_numpy(valid_tensor).float()

        
        if args.subanalysis_type == "one_sample":
               x_train = train_data[4:4+args.n_train,...].permute(0,2,1)  ####  CHANGE HERE
        else:
               x_train = train_data[:args.n_train,...].permute(0,2,1)


        x_test = test_data[:args.n_test,...].permute(0,2,1)
        x_valid = valid_data[:args.n_test,...].permute(0,2,1)
        
        #import pdb; pdb.set_trace()
        args.t_resolution =  x_train.shape[2]
        args.x_resolution =  x_train.shape[1]
        
        if args.t_resolution_train == None:
                args.t_resolution_train = x_train.shape[2]

        if args.t_resolution_test == None:
                args.t_resolution_test =  x_test.shape[2]

        if args.t_resolution_valid == None:
                args.t_resolution_valid = x_valid.shape[2]

        #args.timestamps = [i for i in range(args.t_resolution)]
        args.timestamps_valid = [i for i in range(args.t_resolution_valid)]
        args.timestamps_test = [i for i in range(args.t_resolution_test)]
        args.timestamps = [i*0.004 for i in range(args.t_resolution)]


        res = x_train.shape[1]
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,x_train, x_train, x_train), batch_size=args.batch_size_train, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, x_test, x_test), batch_size=args.batch_size_test, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, x_valid, x_valid, x_valid), batch_size=args.batch_size_test, shuffle=False)

        #data = {"train_loader":train_loader, "test_loader": test_loader, "timestamps":timestamps}
        #import pdb; pdb.set_trace()
        return train_loader,valid_loader, test_loader


def load_dataset_E2(args):

        hdf5_train_file = h5py.File(args.dataset_train_path, 'r')
        hdf5_test_file = h5py.File(args.dataset_test_path, 'r')
        hdf5_valid_file = h5py.File(args.dataset_valid_path, 'r')

        train_loaded_data = hdf5_train_file['train']['pde_250-100'][:]
        test_loaded_data = hdf5_test_file['test']['pde_250-100'][:]
        valid_loaded_data = hdf5_valid_file['valid']['pde_250-100'][:]

        train_tensor =  train_loaded_data.squeeze()
        train_data = torch.from_numpy(train_tensor).float()

        test_tensor =  test_loaded_data.squeeze()
        test_data = torch.from_numpy(test_tensor).float()

        valid_tensor =  valid_loaded_data.squeeze()
        valid_data = torch.from_numpy(valid_tensor).float()


        x_train = train_data[:args.n_train,...].permute(0,2,1)
        x_test = test_data[:args.n_test,...].permute(0,2,1)
        x_valid = valid_data[:args.n_test,...].permute(0,2,1)

        #import pdb; pdb.set_trace()

        beta_min = 0
        beta_max = 0.4

        alpha_train = hdf5_train_file['train']['alpha'][:]
        alpha_train = torch.from_numpy(alpha_train).float().unsqueeze(-1)
        alpha_train_norm = torch.zeros_like(alpha_train)

        beta_train = hdf5_train_file['train']['beta'][:]
        beta_train = torch.from_numpy(beta_train).float().unsqueeze(-1)
        beta_train_norm = (beta_train - beta_min)/(beta_max - beta_min)

        gamma_train = hdf5_train_file['train']['gamma'][:]
        gamma_train = torch.from_numpy(gamma_train).float().unsqueeze(-1)
        gamma_train_norm =  torch.zeros_like(gamma_train)

        parameters_train_tensor = torch.zeros_like(x_train)
        parameter_train = torch.cat((alpha_train_norm, beta_train_norm, gamma_train_norm), dim=-1)[:args.n_train,...]
        parameters_train_tensor[...,:parameter_train.shape[-1]] = parameter_train.unsqueeze(dim=1).repeat(1, 100, 1)



        alpha_valid = hdf5_valid_file['valid']['alpha'][:]
        alpha_valid = torch.from_numpy(alpha_valid).float().unsqueeze(-1)
        alpha_valid_norm = torch.zeros_like(alpha_valid)

        beta_valid = hdf5_valid_file['valid']['beta'][:]
        beta_valid = torch.from_numpy(beta_valid).float().unsqueeze(-1)
        beta_valid_norm = (beta_valid - beta_min)/(beta_max - beta_min)

        gamma_valid = hdf5_valid_file['valid']['gamma'][:]
        gamma_valid = torch.from_numpy(gamma_valid).float().unsqueeze(-1)
        gamma_valid_norm =  torch.zeros_like(gamma_valid)

        parameters_valid_tensor = torch.zeros_like(x_valid)
        parameter_valid = torch.cat((alpha_valid_norm, beta_valid_norm, gamma_valid_norm), dim=-1)[:args.n_test,...]
        parameters_valid_tensor[...,:parameter_valid.shape[-1]] = parameter_valid.unsqueeze(dim=1).repeat(1, 100, 1)




        alpha_test = hdf5_test_file['test']['alpha'][:]
        alpha_test = torch.from_numpy(alpha_test).float().unsqueeze(-1)
        alpha_test_norm = torch.zeros_like(alpha_test)

        beta_test = hdf5_test_file['test']['beta'][:]
        beta_test = torch.from_numpy(beta_test).float().unsqueeze(-1)
        beta_test_norm = (beta_test - beta_min)/(beta_max - beta_min)

        gamma_test = hdf5_test_file['test']['gamma'][:]
        gamma_test = torch.from_numpy(gamma_test).float().unsqueeze(-1)
        gamma_test_norm =  torch.zeros_like(gamma_test)

        parameters_test_tensor = torch.zeros_like(x_test)
        parameter_test = torch.cat((alpha_test_norm, beta_test_norm, gamma_test_norm), dim=-1)[:args.n_test,...]
        parameters_test_tensor[...,:parameter_test.shape[-1]] = parameter_test.unsqueeze(dim=1).repeat(1, 100, 1)



        args.total_t_range = 250
        args.time_stamps = [i*0.004 for i in range(0,args.total_t_range)]

        res = x_train.shape[1]

        #import pdb; pdb.set_trace()
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,x_train, x_train, parameters_train_tensor), batch_size=args.batch_size_train, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, x_test, parameters_test_tensor), batch_size=args.batch_size_test, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, x_valid, x_valid, parameters_valid_tensor), batch_size=args.batch_size_test, shuffle=False)

        #data = {"train_loader":train_loader, "test_loader": test_loader, "timestamps":timestamps}
        #import pdb; pdb.set_trace()
        return train_loader,valid_loader, test_loader













def load_dataset_KS1(args):
        """
        validation is training length  and test is test resolution
        change the file location in arguments file.

        """

        hdf5_train_file = h5py.File(args.dataset_train_path, 'r')
        hdf5_test_file = h5py.File(args.dataset_test_path, 'r')
        hdf5_valid_file = h5py.File(args.dataset_valid_path, 'r')

        L_min = 57.6
        L_max = 70.40

        dt_min = 0.18
        dt_max = 0.22


        train_loaded_data = hdf5_train_file['train']['pde_640-256'][:][:args.n_train, :args.t_resolution_train, :]
        L_train = hdf5_train_file['train']['x'][:][:,-1][:args.n_train]
        dt_train = hdf5_train_file["train"]["dt"][:][:args.n_train]


        test_loaded_data = hdf5_test_file['test']['pde_640-256'][:][:args.n_test, :args.t_resolution_test, :]
        L_test = hdf5_test_file['test']['x'][:][:,-1][:args.n_test]
        dt_test = hdf5_test_file["test"]["dt"][:][:args.n_test]


        valid_loaded_data = hdf5_valid_file['test']['pde_640-256'][:][-args.n_test:, :args.t_resolution_valid, :]
        L_valid = hdf5_valid_file['test']['x'][:][:,-1][-args.n_test:]
        dt_valid = hdf5_valid_file["test"]["dt"][:][-args.n_test:]


        if args.normalise_parameters:
                L_train = (L_train - L_min)/(L_max - L_min)
                dt_train = (dt_train - dt_min)/(dt_max - dt_min)

                L_test = (L_test - L_min)/(L_max - L_min)
                dt_test = (dt_test - dt_min)/(dt_max - dt_min)

                L_valid = (L_valid - L_min)/(L_max - L_min)
                dt_valid = (dt_valid - dt_min)/(dt_max - dt_min)



        x_train = torch.from_numpy(train_loaded_data.squeeze()).float().permute(0,2,1)
        L_train =  torch.from_numpy(L_train).float().unsqueeze(-1)
        dt_train = torch.from_numpy(dt_train).float().unsqueeze(-1)


        x_test = torch.from_numpy(test_loaded_data.squeeze()).float().permute(0,2,1)
        L_test =  torch.from_numpy(L_test).float().unsqueeze(-1)
        dt_test = torch.from_numpy(dt_test).float().unsqueeze(-1)


        x_valid = torch.from_numpy(valid_loaded_data.squeeze()).float().permute(0,2,1)
        L_valid =  torch.from_numpy(L_valid).float().unsqueeze(-1)
        dt_valid = torch.from_numpy(dt_valid).float().unsqueeze(-1)


        # if args.normalise_parameters:
        #         L = torch.cat((L_train, L_test, L_valid), dim=0)
        #         L = normalizer(L.unsqueeze(-1)).squeeze(-1)

        #         L_train = L[:args.n_train]
        #         L_test = L[args.n_train:args.n_train+args.n_test]
        #         L_valid = L[args.n_train+args.n_test:]

        #         dt =torch.cat((dt_train,dt_test,dt_valid), dim=0)
        #         dt = normalizer(dt.unsqueeze(-1)).squeeze(-1)

        #         dt_train = dt[:args.n_train]
        #         dt_test = dt[args.n_train:args.n_train+args.n_test]
        #         dt_valid = dt[args.n_train+args.n_test:]


        parameters_train_tensor = torch.zeros_like(x_train)
        parameter_train = torch.cat((L_train, dt_train), dim=-1)
        parameters_train_tensor[...,:parameter_train.shape[-1]] = parameter_train.unsqueeze(dim=1).repeat(1, x_train.shape[1], 1)



        parameters_test_tensor = torch.zeros_like(x_test)
        parameter_test = torch.cat((L_test, dt_test), dim=-1)
        parameters_test_tensor[...,:parameter_test.shape[-1]] = parameter_test.unsqueeze(dim=1).repeat(1, x_test.shape[1], 1)



        parameters_valid_tensor = torch.zeros_like(x_valid)
        parameter_valid = torch.cat((L_valid, dt_valid), dim=-1)
        parameters_valid_tensor[...,:parameter_valid.shape[-1]] = parameter_valid.unsqueeze(dim=1).repeat(1, x_valid.shape[1], 1)



        #import pdb; pdb.set_trace()
        args.t_resolution =  x_train.shape[2]
        args.x_resolution =  x_train.shape[1]

        args.timestamps = [i for i in range(args.t_resolution)]
        args.timestamps_valid = [i for i in range(args.t_resolution_valid)]
        args.timestamps_test = [i for i in range(args.t_resolution_test)]

        res = x_train.shape[1]

        print("train, test, valid -->", x_train.shape, x_test.shape, x_valid.shape)
        print("parameters:  MAX(train), MIN(test), MAX(valid) -->", torch.max(parameters_train_tensor), torch.min(parameters_test_tensor), torch.max(parameters_valid_tensor))

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,x_train, x_train, parameters_train_tensor), batch_size=args.batch_size_train, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, x_test, parameters_test_tensor), batch_size=args.batch_size_test, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, x_valid, x_valid, parameters_valid_tensor), batch_size=args.batch_size_test, shuffle=False)

        #data = {"train_loader":train_loader, "test_loader": test_loader, "timestamps":timestamps}
        #import pdb; pdb.set_trace()
        return train_loader,valid_loader, test_loader









def load_dataset_KdV(args):
        """
        validation is training length  and test is test resolution
        change the file location in arguments file.

        """

        hdf5_train_file = h5py.File(args.dataset_train_path, 'r')
        hdf5_test_file = h5py.File(args.dataset_test_path, 'r')
        hdf5_valid_file = h5py.File(args.dataset_valid_path, 'r')

        L_min = 141.0
        L_max = 115.0

        dt_min = 0.18
        dt_max = 0.22


        train_loaded_data = hdf5_train_file['train']['pde_640-256'][:][:args.n_train, :args.t_resolution_train, :]
        L_train = hdf5_train_file['train']['x'][:][:,-1][:args.n_train]
        dt_train = hdf5_train_file["train"]["dt"][:][:args.n_train]


        test_loaded_data = hdf5_test_file['test']['pde_640-256'][:][:args.n_test, :args.t_resolution_test, :]
        L_test = hdf5_test_file['test']['x'][:][:,-1][:args.n_test]
        dt_test = hdf5_test_file["test"]["dt"][:][:args.n_test]


        valid_loaded_data = hdf5_valid_file['valid']['pde_640-256'][:][-args.n_test:, :args.t_resolution_valid, :]
        L_valid = hdf5_valid_file['valid']['x'][:][:,-1][:args.n_test]
        dt_valid = hdf5_valid_file["valid"]["dt"][:][:args.n_test]


        if args.normalise_parameters:
                L_train = (L_train - L_min)/(L_max - L_min)
                dt_train = (dt_train - dt_min)/(dt_max - dt_min)

                L_test = (L_test - L_min)/(L_max - L_min)
                dt_test = (dt_test - dt_min)/(dt_max - dt_min)

                L_valid = (L_valid - L_min)/(L_max - L_min)
                dt_valid = (dt_valid - dt_min)/(dt_max - dt_min)



        x_train = torch.from_numpy(train_loaded_data.squeeze()).float().permute(0,2,1)
        L_train =  torch.from_numpy(L_train).float().unsqueeze(-1)
        dt_train = torch.from_numpy(dt_train).float().unsqueeze(-1)


        x_test = torch.from_numpy(test_loaded_data.squeeze()).float().permute(0,2,1)
        L_test =  torch.from_numpy(L_test).float().unsqueeze(-1)
        dt_test = torch.from_numpy(dt_test).float().unsqueeze(-1)


        x_valid = torch.from_numpy(valid_loaded_data.squeeze()).float().permute(0,2,1)
        L_valid =  torch.from_numpy(L_valid).float().unsqueeze(-1)
        dt_valid = torch.from_numpy(dt_valid).float().unsqueeze(-1)


        # if args.normalise_parameters:
        #         L = torch.cat((L_train, L_test, L_valid), dim=0)
        #         L = normalizer(L.unsqueeze(-1)).squeeze(-1)

        #         L_train = L[:args.n_train]
        #         L_test = L[args.n_train:args.n_train+args.n_test]
        #         L_valid = L[args.n_train+args.n_test:]

        #         dt =torch.cat((dt_train,dt_test,dt_valid), dim=0)
        #         dt = normalizer(dt.unsqueeze(-1)).squeeze(-1)

        #         dt_train = dt[:args.n_train]
        #         dt_test = dt[args.n_train:args.n_train+args.n_test]
        #         dt_valid = dt[args.n_train+args.n_test:]


        parameters_train_tensor = torch.zeros_like(x_train)
        parameter_train = torch.cat((L_train, dt_train), dim=-1)
        parameters_train_tensor[...,:parameter_train.shape[-1]] = parameter_train.unsqueeze(dim=1).repeat(1, x_train.shape[1], 1)



        parameters_test_tensor = torch.zeros_like(x_test)
        parameter_test = torch.cat((L_test, dt_test), dim=-1)
        parameters_test_tensor[...,:parameter_test.shape[-1]] = parameter_test.unsqueeze(dim=1).repeat(1, x_test.shape[1], 1)



        parameters_valid_tensor = torch.zeros_like(x_valid)
        parameter_valid = torch.cat((L_valid, dt_valid), dim=-1)
        parameters_valid_tensor[...,:parameter_valid.shape[-1]] = parameter_valid.unsqueeze(dim=1).repeat(1, x_valid.shape[1], 1)



        #import pdb; pdb.set_trace()
        args.t_resolution =  x_train.shape[2]
        args.x_resolution =  x_train.shape[1]

        args.timestamps = [i for i in range(args.t_resolution)]
        args.timestamps_valid = [i for i in range(args.t_resolution_valid)]
        args.timestamps_test = [i for i in range(args.t_resolution_test)]

        res = x_train.shape[1]

        print("train, test, valid -->", x_train.shape, x_test.shape, x_valid.shape)
        print("parameters:  MAX(train), MIN(test), MAX(valid) -->", torch.max(parameters_train_tensor), torch.min(parameters_test_tensor), torch.max(parameters_valid_tensor))

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,x_train, x_train, parameters_train_tensor), batch_size=args.batch_size_train, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, x_test, parameters_test_tensor), batch_size=args.batch_size_test, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, x_valid, x_valid, parameters_valid_tensor), batch_size=args.batch_size_test, shuffle=False)

        #data = {"train_loader":train_loader, "test_loader": test_loader, "timestamps":timestamps}
        #import pdb; pdb.set_trace()
        return train_loader,valid_loader, test_loader
