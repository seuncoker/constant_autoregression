import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
#import matplotlib.pyplot as plt


#from termcolor import colored
import sys, os
from datetime import datetime

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from typing import List

#import csv
import h5py
import argparse

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#import pdb; pdb.set_trace()

from constant_autoregression.argparser import arg_parse 
from constant_autoregression.dataset.load_dataset import load_data, load_dataset_A1, load_dataset_B1, load_dataset_E1, load_dataset_E2, load_dataset_KS1, load_dataset_KdV
from constant_autoregression.util import LpLoss, Printer, get_time, count_params, set_seed, create_current_results_folder, load_auguments, save_config_file
from constant_autoregression.model.load_model import  load_model, get_model
from constant_autoregression.train import training_protocol
from constant_autoregression.test import constant_rollout_test, constant_one_to_one_test, variable_rollout_test, variable_one_to_one_test




################################################################
# load arguments
################################################################

# print("starting.... arguments........")
# parser_int = argparse.ArgumentParser(description="Your script description")
# print("starting.... arguments........")
# # Add arguments with options and help text
# parser_int.add_argument("--argument_name", type=str,  help=" which argument to use ")
# print("starting.... arguments........")
# # Parse arguments from command line
# args_init = parser_int.parse_args()
# print("starting.... arguments........")
# # Access arguments using their names
# arg_name = args_init.argument_name
# print("starting.... arguments........")
# print(arg_name)

# print(f"arg name: {arg_name}")

#import pdb; pdb.set_trace()


################################################################
# load arguments
################################################################

p = Printer(n_digits=6)
args = arg_parse()

p.print("STARTING............................................")

p.print(f"args mode: {args.mode}")


#arg_name =  "arguments"
#arg_name =  "arguments_test"

arg_name = args.argument_file
p.print(f"arg_name: {arg_name}")
args = load_auguments(args, arg_name)

p.print(args)

################################################################
# load seed
################################################################

set_seed(args.seed)


################################################################
# Load training data
################################################################

if args.dataset_name.endswith("B1"):
    args.dataset_train_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/B1/1D_Burgers_Sols_Nu0.01_K1_N2_Sa2500.npy"
    args.dataset_valid_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/B1/1D_Burgers_Sols_Nu0.01_K1_N2_Sa2500.npy"
    args.dataset_test_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/B1/1D_Burgers_Sols_Nu0.01_K1_N2_Sa2500.npy"
    train_loader, val_loader, test_loader = load_dataset_B1(args)


elif args.dataset_name.endswith("A1"):
    args.dataset_train_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/A1/1D_Advection_Sols_beta0.5_K1_N2_Sa2500.npy"
    args.dataset_valid_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/A1/1D_Advection_Sols_beta0.5_K1_N2_Sa2500.npy"
    args.dataset_test_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/A1/1D_Advection_Sols_beta0.5_K1_N2_Sa2500.npy"
    train_loader, val_loader, test_loader = load_dataset_A1(args)

elif args.dataset_name.endswith("E1"):
    args.dataset_train_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/E1/CE_train_E1.h5"
    args.dataset_valid_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/E1/CE_valid_E1.h5"
    args.dataset_test_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/E1/CE_test_E1.h5"
    train_loader, val_loader, test_loader = load_dataset_E1(args)

elif args.dataset_name.endswith("E2"):
    args.dataset_train_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/E1/CE_train_E2.h5"
    args.dataset_valid_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/E1/CE_valid_E2.h5"
    args.dataset_test_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/E1/CE_test_E2.h5"
    train_loader, val_loader, test_loader = load_dataset_E2(args)

elif args.dataset_name.endswith("KS1"):
    args.dataset_train_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/KS1/KS1_train.h5"
    args.dataset_valid_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/KS1/KS1_test.h5"
    args.dataset_test_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/KS1/KS1_test.h5"
    train_loader, val_loader, test_loader = load_dataset_KS1(args)

elif args.dataset_name.endswith("KdV"):
    args.dataset_train_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/KdV/KdV_train_2048.h5"
    args.dataset_valid_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/KdV/KdV_valid.h5"
    args.dataset_test_path = "/mnt/scratch/scoc/constant_autoregression/dataset/data/KdV/KdV_test.h5"
    train_loader, val_loader, test_loader = load_dataset_KdV(args)
else:
    raise TypeError("Specify correct dataset")

p.print(f"Minibatches for train: {len(train_loader)}")
p.print(f"Minibatches for val: {len(val_loader)}")
p.print(f"Minibatches for test: {len(test_loader)}")




mode = args.mode


################################################################
# Test
################################################################


if mode.startswith("test"):

    test_only_path = args.test_only_path 
    test_only_protocol_no = args.test_only_protocol_no
    test_only_epoch_index = args.test_only_epoch_index

    arguments_file = os.path.join(test_only_path, "config" )
    p.print(f"arguments_file:  {arguments_file}")
    args = load_auguments(args, arguments_file)

    p.print(f"args: {args}")

    args.mode = mode


    current_result_save_path = args.current_result_save_path
    experiment = args.experiment
    

    file_saved = "protocol_" + str(test_only_protocol_no) +".pt"
    p.print(f"path --> {os.path.join(test_only_path, file_saved)}" )
    saved_result = torch.load(   os.path.join(test_only_path, file_saved  ),  map_location=device )   
    
    epoch = saved_result["saved_epoch"][test_only_epoch_index]["epoch"]
    last_model_dict = saved_result["saved_epoch"][test_only_epoch_index]["model"]

    model = load_model(args, last_model_dict, device)



    timestamps_train = torch.tensor(args.timestamps_train).to(device)
    timestamps_test = torch.tensor(args.timestamps_test).to(device)
    timestamps_valid = torch.tensor(args.timestamps_valid).to(device)


    p.print(f"timestamps_train: {timestamps_train[:10]}" )
    p.print(f"timestamps_test: {timestamps_test[:10]}" )
    p.print(f"args.t_resolution_train: {args.t_resolution_train}")
    p.print(f"args.t_resolution_test: {args.t_resolution_test}")

    norm = args.norm[0]
    p.print(f"norm: {norm}")

    #p.print("Done..............")
    
    train_cons_ro_250 = constant_rollout_test( args, model, train_loader, timestamps_train, dt_step = args.dt_step, t_resolution=args.t_resolution_train, norm=norm     )
    train_cons_oto_250 = 0 #constant_one_to_one_test( args, model, train_loader, timestamps_train, dt_step = args.dt_step, t_resolution=args.t_resolution_train, norm=norm   )

    valid_cons_ro_250 = 0 #constant_rollout_test( args, model, val_loader, timestamps_valid, dt_step = args.dt_step, t_resolution=args.t_resolution_valid, norm=norm   )
    valid_cons_oto_250 = 0 #constant_one_to_one_test( args, model, val_loader, timestamps_valid, dt_step = args.dt_step, t_resolution=args.t_resolution_valid, norm=norm   )

    test_cons_ro_250 = constant_rollout_test(  args, model, test_loader, timestamps_test, dt_step = args.dt_step, t_resolution=args.t_resolution_test, norm=norm    )
    test_cons_oto_250 = 0 #constant_one_to_one_test(  args, model, test_loader, timestamps_test, dt_step = args.dt_step, t_resolution=args.t_resolution_test, norm=norm    )


    train_var_ro_250 = 0 # variable_rollout_test( args, model, train_loader, timestamps_train, dt_step = args.dt_step, t_resolution=args.t_resolution_train, norm=norm, no_of_steps=100    )
    train_var_oto_250 = 0 #variable_one_to_one_test( args, model, train_loader, timestamps_train, dt_step = args.dt_step, t_resolution=args.t_resolution_train, norm=norm, no_of_steps=105 )

    valid_var_ro_250 = 0 #variable_rollout_test( args, model, val_loader, timestamps_valid, dt_step = args.dt_step, t_resolution=args.t_resolution_valid, norm=norm, no_of_steps=100 )
    valid_var_oto_250 = 0 #variable_one_to_one_test( args, model, val_loader, timestamps_valid, dt_step = args.dt_step, t_resolution=args.t_resolution_valid, norm=norm, no_of_steps=105)

    test_var_ro_250 = 0 #variable_rollout_test(  args, model, test_loader, timestamps_test, dt_step = args.dt_step, t_resolution=args.t_resolution_test, norm=norm, no_of_steps=100 )
    test_var_oto_250 =  0 #variable_one_to_one_test(  args, model, test_loader, timestamps_test, dt_step = args.dt_step, t_resolution=args.t_resolution_test, norm=norm, no_of_steps=105 )
    
    result = {"train_cons_oto_250":train_cons_oto_250, "train_cons_ro_250":train_cons_ro_250, "test_cons_oto_250":test_cons_oto_250, "test_cons_ro_250": test_cons_ro_250, "valid_cons_oto_250":valid_cons_oto_250, "valid_cons_ro_250": valid_cons_ro_250,
              "train_var_oto_250":train_var_oto_250, "train_var_ro_250":train_var_ro_250, "test_var_oto_250":test_var_oto_250, "test_var_ro_250": test_var_ro_250, "valid_var_oto_250":valid_var_oto_250, "valid_var_ro_250": valid_var_ro_250
              }

    p.print(f"save_location: {os.path.join(test_only_path, experiment)}" )
    torch.save(result, os.path.join(test_only_path, experiment + "_result_.pt"))







################################################################
# Training Protocol
################################################################

if mode.startswith("train"):
    #import pdb; pdb.set_trace()



    ################################################################
    # Create result files
    ################################################################

    create_result_folder = True
    if create_result_folder:
        create_current_results_folder(args)



    ################################################################
    # Sort arguments for different training protocol
    ################################################################

    args.number_of_training_protocol = len(args.training_protocols)
    key_names = list(args.training_protocols[0].keys())
    for k in range(len(key_names) ):
        key_name = key_names[k]
        values = [d[key_name] for d in args.training_protocols]
        setattr(args, key_name, values)


    ################################################################
    # Model 
    ################################################################
    #import pdb; pdb.set_trace()
    if args.new_training == False:
        
        load_filename =  torch.load(args.pretrained_model, map_location=device)
        saved_epochs = load_filename["saved_epoch"].copy()
        all_epoch_errors = load_filename["all_error"].clone()
        last_model_dict = saved_epochs[-1]["model"] ### change to -1
        model = load_model(args, last_model_dict, device)
        p.print(f"Continue Training")

    else:
        saved_epochs = None
        all_epoch_errors = None
        p.print(f"New Training")
        model = get_model(args, device)

    p.print(f"model {model}")

    p.print(f"Number of model parameters_complex_function: {count_params(model)}")
    p.print(f"Number of model parameters_numbers: {sum(p.numel() for p in model.parameters())}")

    #import pdb; pdb.set_trace()

    count_t_iter = 0


    timestamps = torch.tensor(args.timestamps_train).to(device)
    #t_iteration = args.iter_per_epochs

    #import pdb; pdb.set_trace()
    p.print(args)


    for proto in range(args.number_of_training_protocol):

        if args.optimiser_type[proto] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[proto], weight_decay=args.weight_decay[proto])
        elif args.optimiser_type[proto] == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate[proto], weight_decay=args.weight_decay[proto])
        
        if args.sheduler_type[proto] == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.sheduler_step[proto], gamma=args.sheduler_gamma[proto])
        elif args.sheduler_type[proto] == "cosine_annealing":
            if args.sheduler_change[proto] == "epoch":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.cos_anneal_T_max[proto]), eta_min=args.min_learning_rate[proto])
            elif args.sheduler_change[proto] == "iteration":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.cos_anneal_T_max[proto]), eta_min=args.min_learning_rate[proto])
        
        if args.new_training == False and args.initialise_optimiser[proto] == False:
            p.print("Loading optimizer last state")
            optimizer.load_state_dict(saved_epochs[-1]["optimizer"])
        else:
            pass


        if args.loss_train_type.startswith("l2"):
            criterion = LpLoss(size_average=False, reduction=False)
            #criterion = LpLoss(size_average=False, reduction=True)
        elif args.loss_train_type.startswith("mse"):
            criterion = torch.nn.MSELoss(reduction="none")
        else:
            pass
        

        #import pdb; pdb.set_trace()
        assert( [ i <= (args.t_resolution_train - args.output_time_stamps)//args.output_time_stamps for i in args.horizon[proto] ] )
        
        max_horizon = round(( args.t_resolution_train - args.input_time_stamps)//args.output_time_stamps)
        p.print(f"max_horizon: {max_horizon}")
        
        p.print(f"dynamic_loss_weight_per_fpass: {args.dynamic_loss_weight_per_fpass[proto]}")

        if args.training_loop == "scheduled_weighted_loss_curriculum":
            args.dynamic_loss_weight_per_fpass[proto] = True

        p.print(f"dynamic_loss_weight_per_fpass: {args.dynamic_loss_weight_per_fpass[proto]}")

        if args.dynamic_loss_weight_per_fpass[proto]:
            ## E1 B1

            args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.45 #5.0056*(max_horizon**(-0.723))

            # if max_horizon == 9 or max_horizon == 10 or max_horizon == 8:   #for output stamps = 25
            #     args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.95 
            # elif max_horizon == 24 or max_horizon == 25 or max_horizon == 23:   #for output stamps = 10
            #     args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.5
            # elif max_horizon == 19 or max_horizon == 20 or max_horizon == 21:   #for output stamps = 10
            #     args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.47
            # elif max_horizon == 39 or max_horizon == 40 or max_horizon == 41:   #for output stamps = 10
            #     args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.39
            # elif max_horizon == 49 or max_horizon == 50 or max_horizon == 48:   #for output stamps = 10
            #     args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.3125
            # elif max_horizon == 249 or max_horizon == 250 or max_horizon == 248:   #for output stamps = 1
            #     args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.125 #0.05 #0.1 #0.075 #0.1
            
            # #### KS
            # elif max_horizon == 139 or max_horizon == 140 or max_horizon == 138:   #for output stamps = 1
            #     args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.13
            # elif max_horizon == 639:
            #     args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.038

            # #### KdV
            # elif max_horizon == 62 or max_horizon == 63 or max_horizon == 64:   #for output stamps = 1
            #     args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.13
            # else:
            #     raise TypeError("the selected output_timestamps to determines the maximum horizon is not among the dynamic weight constant specified")

            #p.print(f" maximum horizon: {args.dynamic_loss_weight_per_fpass_constant_parameter[proto]}")
            p.print(f" dynamic weighting parameter: {args.dynamic_loss_weight_per_fpass_constant_parameter[proto]}")
        
        
        p.print(f"scheduler_change: {args.sheduler_change[proto]}")

        #import pdb; pdb.set_trace()
        saved_results, count_t_iter = training_protocol(
            proto = proto,
            args = args,
            epochs = args.epochs[proto],
            model = model,
            optimizer = optimizer,
            scheduler = scheduler,
            train_loader = train_loader,
            valid_loader = val_loader,
            test_loader = test_loader,
            saved_epochs = saved_epochs,
            all_epoch_errors = all_epoch_errors,
            criterion = criterion,
            count_t_iter = count_t_iter,
            
            dt_step = args.dt_step,
            input_time_stamps = args.input_time_stamps,
            output_time_stamps = args.output_time_stamps,
            t_resolution = args.t_resolution_train,


            max_horizon = max_horizon,
            timestamps = timestamps,
            horizon = args.horizon[proto],
            random_horizon = args.random_horizon[proto],
            
            t_iteration =  args.iter_per_epochs[proto],

            sheduler_change = args.sheduler_change[proto],

            dynamic_loss_weight_per_fpass = args.dynamic_loss_weight_per_fpass[proto],
            dynamic_loss_weight_per_fpass_type = args.dynamic_loss_weight_per_fpass_type[proto],
            dynamic_loss_weight_per_fpass_reversed = args.dynamic_loss_weight_per_fpass_reversed[proto],
            dynamic_loss_weight_per_fpass_constant_parameter = args.dynamic_loss_weight_per_fpass_constant_parameter[proto],


            dynamic_loss_weight_per_tstamp = args.dynamic_loss_weight_per_tstamp[proto],
            dynamic_loss_weight_per_tstamp_constant_parameter = args.dynamic_loss_weight_per_tstamp_constant_parameter[proto],

            noise = args.noise[proto],
            noise_std = args.noise_std[proto],
            norm = args.norm[proto],

            push_forward = args.push_forward[proto],
            push_forward_parameter_random = args.push_forward_parameter_random[proto],
            push_forward_parameter = args.push_forward_parameter[proto],

            )
        
        saved_epochs = saved_results["saved_epoch"].copy()
        all_epoch_errors = saved_results["all_error"].clone()
        p.print(f"End of Training Protocol {proto+1} ..........")   


# elif args.mode.startswith("test"):
#     ##import pdb; pdb.set_trace()

#     test_only_path = args.test_only_path 
#     test_only_protocol_no = args.test_only_protocol_no
#     test_only_epoch_index = args.test_only_epoch_index

#     arguments_file = os.path.join(test_only_path, "config" )
#     args = load_auguments(args, arguments_file)
#     current_result_save_path = args.current_result_save_path
#     experiment = args.experiment
#     timestamps = torch.tensor(args.timestamps).to(device)


#     file_saved = "protocol_" + str(test_only_protocol_no) +".pt"
#     print("path -->", os.path.join(test_only_path, file_saved) )
#     saved_result = torch.load(   os.path.join(test_only_path, file_saved  ),  map_location=device )   
    
#     epoch = saved_result["saved_epoch"][test_only_epoch_index]["epoch"]
#     last_model_dict = saved_result["saved_epoch"][test_only_epoch_index]["model"]
    
#     #last_model_dict = saved_result
    
#     #import pdb; pdb.set_trace()
#     model = load_model(args, last_model_dict, device)


#     cons_ro_250 = constant_rollout_test( args, model, test_loader, timestamps, dt_step = args.dt_step )
#     train_cons_ro_250 = constant_rollout_test( args, model, train_loader, timestamps, dt_step = args.dt_step )

#     #import pdb; pdb.set_trace()
#     cons_oto_250 = constant_one_to_one_test( args, model, test_loader, timestamps, dt_step = args.dt_step)
#     train_cons_oto_250 = constant_one_to_one_test( args, model, train_loader, timestamps, dt_step = args.dt_step )

#     result = {"train_cons_oto_250":train_cons_oto_250, "train_cons_ro_250":train_cons_ro_250, "cons_oto_250":cons_oto_250, "cons_ro_250": cons_ro_250}

#     print(os.path.join(test_only_path, experiment) )
#     torch.save(result, os.path.join(test_only_path, experiment + ".pt") )

