import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt


from termcolor import colored
import sys, os
from datetime import datetime

import operator
from functools import reduce
from functools import partial
from timeit import default_timer

import csv
import h5py

# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))

from variable_autoregression.argparser import arg_parse
from variable_autoregression.training_loop.random_sampling import random_time_sampling, random_time_sampling_new, random_time_sampling_new_new, random_time_sampling_splits, random_time_sampling_one_sample, sequential_time_sampling
from variable_autoregression.training_loop.transformer_loop import run_epoch
from variable_autoregression.util import LpLoss, Printer, get_time, count_params, set_seed, return_checkpoint, dynamic_weight_loss, dynamic_weight_loss_sq, create_current_results_folder, load_auguments, save_config_file, create_data, create_next_data
from variable_autoregression.test import constant_rollout_test, constant_one_to_one_test, variable_rollout_test, variable_one_to_one_test, constant_one_to_one_test_splits



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p = Printer(n_digits=6)
# args = arg_parse()
# args = load_auguments(args, "arguments")










# class TrainState:
#     """Track number of steps, examples, and tokens processed"""

#     step: int = 0  # Steps in the current epoch
#     accum_step: int = 0  # Number of gradient accumulation steps
#     samples: int = 0  # total # of examples used
#     tokens: int = 0 



# ############################################################
# ## Batch Data
# ############################################################



# class Batch:
#     """Object for holding a batch of data with mask during training."""

#     def __init__(self, input, input_time, output, output_time, modes_in, modes_out):  # 2 = <blank>
#         #self.src = src
#         self.input = input
#         self.input_mask = torch.ones((1, 1, 1, input.shape[1]) )
#         #self.input_mask = torch.ones((1, 1, input.shape[1]) )#.to(device)
#         #self.input_mask = torch.ones((1, 1, 100) )#.to(device)
#         self.input_time = input_time
#         self.output_time = output_time

#         self.modes_in = modes_in
#         self.modes_out = modes_out

#         if output is not None:

#             self.output = output
#             self.output_in = torch.cat((input[:,-1:], output[:,:-1]),dim=1)#.to(device)
#             self.output_in_time = torch.cat((input_time[:,-1:], output_time[:,:-1]),dim=1)#.to(device)

#             #self.output_out = output[:,1:]
#             self.output_mask = self.subsequent_mask(self.output)
#             self.ntokens =  1 # (self.tgt_y != pad).data.sum()

#     @staticmethod
#     def subsequent_mask(tgt):
#         "Mask out subsequent positions."
#         size =tgt.shape[1]
#         #size = 100
#         #print("size -->",size)
#         #attn_shape = (1,1, size, size)
#         attn_shape = (1, size, size)
#         subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
#             torch.uint8
#         )#.to(device)
#     #print("subsequent_mask -->", subsequent_mask.shape)
#         return subsequent_mask == 0




# ############################################################
# ## Data Generator
# ############################################################

# def data_gen(tensor, in_ntimes, out_ntimes, batch_size, nbatches, modes_in, modes_out):
#     #tensor = torch.from_numpy(tensor[:batch_size*nbatches]).float()#.to(device)
#     # print("tensor -->", tensor.shape)
#     # print("in_ntimes -->", in_ntimes.shape)
#     # print("out_ntimes -->", out_ntimes.shape)
#     # input_function = torch.zeros((batch_size*nbatches, in_ntimes.shape[1], tensor.shape[2]))#.to(device)
#     # output_function = torch.zeros((batch_size*nbatches, out_ntimes.shape[1], tensor.shape[2]))#.to(device)
#     for i in range(nbatches):
#         start = i*batch_size
#         end = (i+1)*batch_size
#         #print("start, end  -->", start, end)
#         time_in = in_ntimes[start:end,:]
#         time_out = out_ntimes[start:end,:]
#         #print("time_in -->", time_in[0])
#         #print("time_out -->", time_out[0])
#         input_function = torch.gather(tensor[start:end], 1, time_in.unsqueeze(-1).repeat(1,1,200))
#         output_function = torch.gather(tensor[start:end], 1, time_out.unsqueeze(-1).repeat(1,1,200))

#         time_in_t = torch.gather(time_stamps_sub[start:end,:], 1, time_in )
#         time_out_t = torch.gather(time_stamps_sub[start:end,:], 1, time_out )
#         #print("time_in_t -->", time_in_t[0:3])
#         #print("time_out_t -->", time_out_t[0:3])
#         #print("input_function -->", input_function.shape)
#         #print("output_function -->", output_function.shape)
#         #yield Batch(input_function, in_ntimes, output_function, out_ntimes, modes_in, modes_out )

#         yield Batch(input_function, time_in_t, output_function, time_out_t, modes_in, modes_out )


















def training_protocol(
                    proto,
                    args,
                    epochs,
                    model,
                    optimizer,
                    scheduler,
                    train_loader,
                    valid_loader,
                    test_loader,
                    saved_epochs,
                    all_epoch_errors,
                    criterion,
                    count_t_iter,

                    dt_step,
                    input_time_stamps,
                    output_time_stamps,
                    t_resolution,


                    max_horizon,
                    timestamps,
                    horizon,
                    random_horizon,
                    #n_tsamples,
                    t_iteration,

                    sheduler_change,

                    dynamic_loss_weight_per_fpass,
                    dynamic_loss_weight_per_fpass_type,
                    dynamic_loss_weight_per_fpass_reversed,
                    dynamic_loss_weight_per_fpass_constant_parameter,

                    dynamic_loss_weight_per_tstamp,
                    dynamic_loss_weight_per_tstamp_constant_parameter,

                    noise,
                    noise_std,
                    norm,

                    push_forward,
                    push_forward_parameter_random,
                    push_forward_parameter,

                    ):
    
    

    #assert( [ i <= (args.t_resolution - args.output_time_stamps)//args.output_time_stamps for i in args.horizon ] )

    #args.n_tsamples = [ (i+1)*args.output_time_stamps for i in args.horizon]  #########################################
    #args.n_tsamples = [args.input_time_stamps+(i*args.output_time_stamps) for i in args.horizon]
    #t_iteration = args.iter_per_epochs
    #assert args.input_time_stamps == args.next_input_time_stamps
    #import pdb; pdb.set_trace()

    last_epoch_no = 0
    results = []
    last_result = []
    error = torch.zeros(epochs,10).to(device)


    if saved_epochs != None:
        last_epoch_no = saved_epochs[-1]["epoch"]
        results = saved_epochs
        error = torch.cat((all_epoch_errors, error), dim=0)


    #max_horizon = round(( args.t_resolution - args.input_time_stamps)//args.output_time_stamps)
    n_tsamples = [input_time_stamps+(i*output_time_stamps) for i in horizon]

    if args.time_prediction == "constant":
        assert args.time_sampling_choice == 1  # use the right type of random time generator
        assert dt_step-1 <= int( t_resolution/max(n_tsamples) )

    elif args.time_prediction == "variable":
        assert args.time_sampling_choice > 1


    print("max horizon ->", max_horizon)

    f_pass_weights = torch.ones( max_horizon).to(device)
    t_step_weights =  torch.ones( args.output_time_stamps ).to(device)

    # if args.dynamic_loss_weight_per_fpass:
    #     if max_horizon == 9 or max_horizon == 10 or max_horizon == 8:   #for output stamps = 25
    #         args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.95 
    #     elif max_horizon == 24 or max_horizon == 25 or max_horizon == 23:   #for output stamps = 10
    #         args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.5
    #     elif max_horizon == 249 or max_horizon == 250 or max_horizon == 248:   #for output stamps = 1
    #         args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.075
        
    #     #### KS
    #     elif max_horizon == 139 or max_horizon == 140 or max_horizon == 138:   #for output stamps = 1
    #         args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.13
    #     elif max_horizon == 639:
    #         args.dynamic_loss_weight_per_fpass_constant_parameter[proto] = 0.038
    #     else:
    #         raise TypeError("the selected output_timestamps to determines the maximum horizon is not among the dynamic weight constant specified")

    #     print(" dynamic weighting parameter ->", args.dynamic_loss_weight_per_fpass_constant_parameter[proto])
    #count_t_iter = epochs*t_iteration
    #import pdb; pdb.set_trace()
    

    save_config_file(args)
    model.train()
    
    
    p.print(f"Input multistep: {args.input_time_stamps}")
    p.print(f"Output multistep: {args.output_time_stamps}")
    p.print(f"t_resolution_train: {t_resolution}")
    p.print(f"timestamps:{len(timestamps)}")
    p.print(f"horizon: {horizon}")
    p.print(f"Number of tsamples : {n_tsamples}")
    p.print(f"Number of iteration per tsamples: {t_iteration}")
    p.print(f"total iteration: {epochs*t_iteration}")
    p.print(f"noise:  {noise}")
    p.print(f"noise_std: {noise_std}")
    p.print(f"norm: {norm}")
    p.print(f"dt_step: {dt_step}")
    p.print(f"dynamic_loss_weight_per_fpass: {dynamic_loss_weight_per_fpass}")
    p.print(f"dynamic_loss_weight_per_tstamp: {dynamic_loss_weight_per_tstamp}")
    p.print(f"push_forward: {push_forward}")
    p.print(f"scheduler: {scheduler}")
    p.print(f"sheduler_change: {sheduler_change}")
    p.print(f"horizon: {horizon}")
    p.print(f"predict_difference: {args.predict_difference}")

    for ep in range(last_epoch_no+1, last_epoch_no + epochs + 1):

        t1 = default_timer()

        #import pdb; pdb.set_trace()

        if args.training_loop == "random_time_sampling":
            train_l2_full, model, count_t_iter = random_time_sampling_new(

                            args,
                            count_t_iter,
                            proto,

                            ep,
                            epochs,
                            last_epoch_no,
                            
                            t_iteration = t_iteration,
                            n_tsamples = n_tsamples,

                            data_batchsize = args.batch_size_train,

                            model = model,
                            optimizer = optimizer,
                            train_loader = train_loader,
                            criterion = criterion,


                            input_time_stamps = input_time_stamps,
                            output_time_stamps = output_time_stamps,
                            t_resolution = t_resolution,
                            timestamps = timestamps,

                            f_pass_weights = f_pass_weights,
                            t_step_weights = t_step_weights,

                            time_prediction = args.time_prediction,
                            time_conditioning = args.time_conditioning,

                            max_horizon = max_horizon,
                            horizons = horizon,
                            random_horizon = random_horizon,

                            dt_step = dt_step,

                            noise = noise,
                            noise_std = noise_std,
                            norm = norm,

                            scheduler = scheduler,
                            sheduler_change = sheduler_change,

                            dynamic_loss_weight_per_fpass = dynamic_loss_weight_per_fpass,
                            dynamic_loss_weight_per_fpass_type = dynamic_loss_weight_per_fpass_type,
                            dynamic_loss_weight_per_fpass_reversed = dynamic_loss_weight_per_fpass_reversed,
                            dynamic_loss_weight_per_fpass_constant_parameter = dynamic_loss_weight_per_fpass_constant_parameter,

                            dynamic_loss_weight_per_tstamp= dynamic_loss_weight_per_tstamp,
                            dynamic_loss_weight_per_tstamp_constant_parameter = dynamic_loss_weight_per_tstamp_constant_parameter,

                            push_forward = push_forward,
                            push_forward_parameter_random = push_forward_parameter_random,
                            push_forward_parameter = push_forward_parameter,


            )

        elif args.training_loop == "sequential_time_sampling":

            x1,y1 = 1,1
            x2,y2 = epochs, max_horizon
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m*x1
            horizon = [ int(m*ep + b) ]

            #horizon = [max_horizon]


            p.print(f"{ep} ...............................................")
            p.print(f"horizon: {horizon}")
            n_tsamples = [input_time_stamps+(i*output_time_stamps) for i in horizon]
            p.print(f"Number of tsamples : {n_tsamples}")


            train_l2_full, model, count_t_iter = sequential_time_sampling(

                            args,
                            count_t_iter,
                            proto,

                            ep,
                            epochs,
                            last_epoch_no,
                            
                            t_iteration = t_iteration,
                            n_tsamples = n_tsamples,

                            data_batchsize = args.batch_size_train,

                            model = model,
                            optimizer = optimizer,
                            train_loader = train_loader,
                            criterion = criterion,


                            input_time_stamps = input_time_stamps,
                            output_time_stamps = output_time_stamps,
                            t_resolution = t_resolution,
                            timestamps = timestamps,

                            f_pass_weights = f_pass_weights,
                            t_step_weights = t_step_weights,

                            time_prediction = args.time_prediction,
                            time_conditioning = args.time_conditioning,

                            max_horizon = max_horizon,
                            horizons = horizon,
                            random_horizon = random_horizon,

                            dt_step = dt_step,

                            noise = noise,
                            noise_std = noise_std,
                            norm = norm,

                            scheduler = scheduler,
                            sheduler_change = sheduler_change,

                            dynamic_loss_weight_per_fpass = dynamic_loss_weight_per_fpass,
                            dynamic_loss_weight_per_fpass_type = dynamic_loss_weight_per_fpass_type,
                            dynamic_loss_weight_per_fpass_reversed = dynamic_loss_weight_per_fpass_reversed,
                            dynamic_loss_weight_per_fpass_constant_parameter = dynamic_loss_weight_per_fpass_constant_parameter,

                            dynamic_loss_weight_per_tstamp= dynamic_loss_weight_per_tstamp,
                            dynamic_loss_weight_per_tstamp_constant_parameter = dynamic_loss_weight_per_tstamp_constant_parameter,

                            push_forward = push_forward,
                            push_forward_parameter_random = push_forward_parameter_random,
                            push_forward_parameter = push_forward_parameter,

            )

        elif args.training_loop == "transformer_loop":

            train_l2_full, model, count_t_iter = run_epoch(

                            args,
                            count_t_iter,
                            proto,

                            ep,
                            epochs,
                            last_epoch_no,
                            
                            t_iteration = t_iteration,
                            n_tsamples = n_tsamples,

                            data_batchsize = args.batch_size_train,

                            model = model,
                            optimizer = optimizer,
                            train_loader = train_loader,
                            criterion = criterion,


                            input_time_stamps = input_time_stamps,
                            output_time_stamps = output_time_stamps,
                            t_resolution = t_resolution,
                            timestamps = timestamps,

                            f_pass_weights = f_pass_weights,
                            t_step_weights = t_step_weights,

                            time_prediction = args.time_prediction,
                            time_conditioning = args.time_conditioning,

                            max_horizon = max_horizon,
                            horizons = horizon,
                            random_horizon = random_horizon,

                            dt_step = dt_step,

                            noise = noise,
                            noise_std = noise_std,
                            norm = norm,

                            scheduler = scheduler,
                            sheduler_change = sheduler_change,

                            dynamic_loss_weight_per_fpass = dynamic_loss_weight_per_fpass,
                            dynamic_loss_weight_per_fpass_type = dynamic_loss_weight_per_fpass_type,
                            dynamic_loss_weight_per_fpass_reversed = dynamic_loss_weight_per_fpass_reversed,
                            dynamic_loss_weight_per_fpass_constant_parameter = dynamic_loss_weight_per_fpass_constant_parameter,

                            dynamic_loss_weight_per_tstamp= dynamic_loss_weight_per_tstamp,
                            dynamic_loss_weight_per_tstamp_constant_parameter = dynamic_loss_weight_per_tstamp_constant_parameter,

                            push_forward = push_forward,
                            push_forward_parameter_random = push_forward_parameter_random,
                            push_forward_parameter = push_forward_parameter,


            )
        
        else: 
            raise TypeError("Select training protocol type...")  


    
        current_lr = optimizer.param_groups[0]['lr']

        

        if sheduler_change == "epoch":
            scheduler.step()


        timestamps_train = torch.tensor(args.timestamps_train).to(device)
        timestamps_test = torch.tensor(args.timestamps_test).to(device)
        timestamps_valid = torch.tensor(args.timestamps_valid).to(device)

        train_cons_ro_250 = [0,0,0]
        train_cons_oto_250 = [0,0,0]

        train_var_ro_250 = [0,0,0]
        train_var_oto_250 = [0,0,0]

        test_var_ro_250 = [0,0,0]
        test_var_oto_250 = [0, 0, 0]
        
        #import pdb; pdb.set_trace()
        #test_cons_ro_250 = constant_rollout_test(  args, model, test_loader, timestamps_test, dt_step = args.dt_step, t_resolution=args.t_resolution_test, norm = args.norm[proto]    )
        #test_cons_oto_250 = constant_one_to_one_test(  args, model, test_loader, timestamps_test, dt_step = args.dt_step, t_resolution=args.t_resolution_test, norm = args.norm[proto]    )
        test_cons_ro_250 = [0,0,0]
        test_cons_oto_250 = [0,0,0]

        valid_cons_ro_250 = [0,0,0]
        valid_cons_oto_250 = [0,0,0]


        #import pdb; pdb.set_trace()
        if (ep) % args.epoch_print_interval == 0:
            train_cons_ro_250 = constant_rollout_test( args, model, train_loader, timestamps_train, dt_step = args.dt_step, t_resolution=args.t_resolution_train, norm = args.norm[proto]  )
            train_cons_oto_250 = constant_one_to_one_test( args, model, train_loader, timestamps_train, dt_step = args.dt_step, t_resolution=args.t_resolution_train, norm = args.norm[proto]   )

            train_var_ro_250 = variable_rollout_test( args, model, train_loader, timestamps_train, dt_step = args.dt_step, t_resolution=args.t_resolution_train, norm = args.norm[proto]  )
            train_var_oto_250 = variable_one_to_one_test( args, model, train_loader, timestamps_train, dt_step = args.dt_step, t_resolution=args.t_resolution_train, norm = args.norm[proto]   )



            test_cons_ro_250 = constant_rollout_test( args, model, test_loader, timestamps_test, dt_step = args.dt_step, t_resolution=args.t_resolution_test, norm = args.norm[proto]  )
            test_cons_oto_250 = constant_one_to_one_test( args, model, test_loader, timestamps_test, dt_step = args.dt_step, t_resolution=args.t_resolution_test, norm = args.norm[proto]   )

            test_var_ro_250 = variable_rollout_test( args, model, test_loader, timestamps_test, dt_step = args.dt_step, t_resolution=args.t_resolution_test, norm = args.norm[proto]  )
            test_var_oto_250 = variable_one_to_one_test( args, model, test_loader, timestamps_test, dt_step = args.dt_step, t_resolution=args.t_resolution_test, norm = args.norm[proto]   )


            # valid_cons_ro_250 = constant_rollout_test( args, model, valid_loader, timestamps_valid, dt_step = args.dt_step, t_resolution=args.t_resolution_valid   )
            # valid_cons_oto_250 = constant_one_to_one_test( args, model, valid_loader, timestamps_valid, dt_step = args.dt_step, t_resolution=args.t_resolution_valid   )

            # test_cons_ro_250 = constant_rollout_test(  args, model, test_loader, timestamps_test, dt_step = args.dt_step, t_resolution=args.t_resolution_test    )
            # test_cons_oto_250 = constant_one_to_one_test(  args, model, test_loader, timestamps_test, dt_step = args.dt_step, t_resolution=args.t_resolution_test    )
            
            # train_cons_ro_250 = [0,0,0] # constant_one_to_one_test_splits( args, model, train_loader, timestamps, dt_step = args.dt_step )
            # train_cons_oto_250 = constant_one_to_one_test_splits( args, model, train_loader, timestamps, dt_step = args.dt_step )

            # test_cons_ro_250 = [0,0,0] #constant_rollout_test(  args, model, test_loader, timestamps, dt_step = args.dt_step  )
            # test_cons_oto_250 = constant_one_to_one_test_splits(  args, model, test_loader, timestamps, dt_step = args.dt_step  )
            
            # print("prediction ->", test_cons_oto_250[1].squeeze()[::8, ::40])
            # print("true -->", test_cons_oto_250[2].squeeze()[::8, ::40])
            # print("loss -->", test_cons_oto_250[0])
            # print("\n")

        t2 = default_timer()
        
        #p.print(f"TRAIN: ep: {ep}, time: {(t2-t1):.4f}, lr: {current_lr}, TRAIN_CONST_OTO_250: {train_cons_oto_250[0]:.4f}, TRAIN_CONST_RO_250: {train_cons_ro_250[0]:.4f}, TEST_CONST_OTO_250: {test_cons_oto_250[0]:.4f}, TEST_CONST_RO_250: {test_cons_ro_250[0]:.4f}, VALID_CONST_OTO_250: {valid_cons_oto_250[0]:.4f}, VALID_CONST_RO_250: {valid_cons_ro_250[0]:.4f}")
        p.print(f"TRAIN: ep: {ep}, time: {(t2-t1):.4f}, lr: {current_lr}, TRAIN_CONST_OTO_250: {train_cons_oto_250[0]:.4f}, TRAIN_CONST_RO_250: {train_cons_ro_250[0]:.4f}, TRAIN_VAR_OTO_250: {train_var_oto_250[0]:.4f}, TRAIN_VAR_RO_250: {train_var_ro_250[0]:.4f}, TEST_VAR_OTO_250: {test_var_oto_250[0]:.4f}, TEST_VAR_RO_250: {test_var_ro_250[0]:.4f}")


        # var_ro_250 = 0.00000 # variable_rollout_test(  args, model, train_loader, timestamps, tsamples = 250 )
        # var_oto_250 = 0.000000 #variable_one_to_one_test(  args, model, train_loader, timestamps, tsamples = 250 )
        # p.print(f"TRAIN: ep: {ep}, time: {(t2-t1):.4f}, lr: {current_lr}, CONST_OTO_250: {cons_oto_250[0]:.4f}, CONST_RO_250: {cons_ro_250[0]:.4f}, VAR_OTO_250: {var_oto_250[0]:.4f}, VAR_RO_250: {var_ro_250[0]:.4f}")

        ep_error = torch.tensor([ep, t2-t1,current_lr,train_cons_oto_250[0],train_cons_ro_250[0], test_cons_oto_250[0], test_cons_ro_250[0] ]).to(device)
        error[ep-1,:ep_error.shape[0]] = ep_error

        #epoch_info = {"train_oto_250":train_cons_oto_250, "train_ro_250": train_cons_ro_250, "test_oto_250":test_cons_oto_250, "test_ro_250":test_cons_ro_250 }
        epoch_info = {"train_oto_250":train_cons_oto_250, "train_ro_250": train_cons_ro_250, "train_var_oto_250":train_var_oto_250, "train_var_ro_250":train_var_ro_250, "test_var_oto_250":test_var_oto_250, "test_var_ro_250":test_var_ro_250 }
        if (ep) % args.epoch_save_interval == 0:
            results.append(return_checkpoint(ep, t2-t1,current_lr, epoch_info, model, optimizer))



    results.append(return_checkpoint(ep, t2-t1,current_lr, epoch_info, model, optimizer))
    torch.save({"saved_epoch": results, "all_error":error }, os.path.join(args.current_result_save_path, f"protocol_{proto+1}.pt"))
    
    #result = {"train_cons_oto_250":train_cons_oto_250, "train_cons_ro_250":train_cons_ro_250, "cons_oto_250":test_cons_oto_250, "cons_ro_250": test_cons_ro_250, "valid_cons_oto_250":valid_cons_oto_250, "valid_cons_ro_250": valid_cons_ro_250}
    result = {"train_cons_oto_250":train_cons_oto_250, "train_cons_ro_250":train_cons_ro_250, "train_var_oto_250":train_var_oto_250, "train_var_ro_250": train_var_ro_250, "test_var_oto_250":test_var_oto_250, "test_var_ro_250": test_var_ro_250}
    torch.save(result, os.path.join(args.current_result_save_path, f"{args.experiment}_proto_{proto+1}.pt") )

    if proto == args.number_of_training_protocol - 1:
        torch.save(error, os.path.join(args.current_result_save_path, f"errors.pt") )
    
    last_result.append(return_checkpoint(ep, t2-t1, current_lr, epoch_info, model, optimizer))
    

    
    return {"saved_epoch": last_result, "all_error": error}, count_t_iter





