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

#import csv
import h5py


from constant_autoregression.util import LpLoss, Printer, get_time, count_params, set_seed, return_checkpoint, dynamic_weight_loss, dynamic_weight_loss_sq, create_current_results_folder, load_auguments, save_config_file, create_data, create_next_data, batch_time_sampling, train_print_time, Normalizer_1D, k_transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p = Printer(n_digits=6)




normalizer = Normalizer_1D()





def random_time_sampling(
    args,
    count_t_iter,
    proto,

    ep,
    last_epoch_no,

    t_iteration,
    n_tsamples,
    data_batchsize,

    model,
    optimizer,
    train_loader,
    criterion,


    next_input_time_stamps,
    output_time_stamps,
    t_resolution, 

    timestamps,

    f_pass_weights,
    t_step_weights,

    time_prediction,
    time_conditioning,

    horizon,

    dt_step,
    
    norm,
    noise,
    noise_std


    ):


    train_l2_full = 0
    

    #import pdb; pdb.set_trace()

    total_iter = sum(args.epochs)*t_iteration
    t_sample_space = torch.arange(t_resolution)

    if time_prediction == "constant":
        assert args.time_sampling_choice == 1  # use the right type of random time generator
        assert dt_step-1 <= int( t_resolution/max(n_tsamples) )

    # elif time_prediction == "variable":
    #     assert args.time_sampling_choice > 1

    for out_samp in range(len(n_tsamples)):
        assert len(torch.tensor([t for t in range(0, (args.t_resolution -  (out_samp + ((out_samp -1)*(dt_step-1)) ) + 1 ))]) ) > 0 ## check that the length of the initial sample range is positvie 


    for s in range(t_iteration):
        #import pdb; pdb.set_trace()
        count_t_iter += 1

        for out_samp in range(len(n_tsamples)):

            tsamples = n_tsamples[out_samp]
            horizon = args.horizon[out_samp] #(tsamples-output_time_stamps)//output_time_stamps

            # p.print(f"tsample: {tsamples} ")
            # p.print(f"horizon: {horizon}")


            if args.dynamic_loss_weight_per_fpass[proto]:
                f_pass_weights = dynamic_weight_loss_sq(count_t_iter, total_iter, args.dynamic_loss_weight_per_fpass_constant_parameter[proto], (args.t_resolution - args.output_time_stamps)//args.output_time_stamps, args.horizon).to(device)
            elif args.dynamic_loss_weight_per_fpass[proto] == None:
                raise TypeError("Specify dynamic_w_l_f_pass (True or False) ")

            if args.dynamic_loss_weight_per_fpass_reversed[proto]:
                f_pass_weights = dynamic_weight_loss_sq(total_iter- count_t_iter + 1, total_iter,args.dynamic_loss_weight_per_fpass_constant_parameter[proto], (args.t_resolution - args.output_time_stamps)//args.output_time_stamps, args.horizon).to(device)
                f_pass_weights = torch.flip(f_pass_weights, dims=[0])
            elif args.dynamic_loss_weight_per_fpass_reversed[proto] == None:
                raise TypeError("Specify dynamic_w_l_f_pass_reversed (True or False) ")

            if args.dynamic_loss_weight_per_tstamp[proto]:
                t_step_weights = dynamic_weight_loss(count_t_iter, total_iter, args.dynamic_loss_weight_per_tstamp_constant_parameter[proto],  args.output_time_stamps  ).to(device)
            elif args.dynamic_loss_weight_per_tstamp[proto] == None:
                raise TypeError("Specify dynamic_w_l_t_steps (True or False) ")


            #import pdb; pdb.set_trace()

            for (data, u_super, x, parameters) in train_loader:

                #import pdb; pdb.set_trace()
                data = data.to(device)
                parameters = parameters[...,:args.no_parameters].to(device)

                time_sampling_choice = args.time_sampling_choice
                data_batch = batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
                time_indicies = t_sample_space[data_batch.indicies]
                xy = torch.gather(data, -1, time_indicies.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
                xy_t = torch.ones_like(xy)[:,0,:].to(device)
                xy_t = xy_t*timestamps[time_indicies]
                xy_t = xy_t.unsqueeze(1).repeat(1,data.shape[1],1)
                xy_tindicies = time_indicies.long()

  
                time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, next_input_time_stamps)]
                
                x = xy[..., :output_time_stamps ]
                x_t = xy_t[..., :output_time_stamps ]
                x_tindicies = xy_tindicies[..., :output_time_stamps ]


                
                loss = 0
                a_l = 0


                #import pdb; pdb.set_trace()
                f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)

                if args.dynamic_loss_weight_per_fpass[proto]:
                    
                    # if args.dynamic_loss_weight_per_fpass_type.startswith("global"):
                    #     random_steps = time_indicies[:,output_time_stamps]
                    #     f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
                    #     random_steps_dx = [ torch.div(irx, output_time_stamps, rounding_mode='floor')-1 for irx in random_steps]
                    #     for irx in range(len(random_steps_dx)):
                    #         f_pass_weights_random[irx,:] = torch.tensor([f_pass_weights[irx].item() for irx in range(random_steps_dx[irx],random_steps_dx[irx]+horizon)]).to(device)

                    if args.dynamic_loss_weight_per_fpass_type.startswith("global"):
                        random_steps = time_indicies[:,output_time_stamps:][:,::output_time_stamps]
                        f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
                        for irx in range(args.batch_size_train):
                            random_steps_dx = [ torch.div(irx-output_time_stamps, output_time_stamps, rounding_mode='floor').item() for irx in random_steps[irx]]
                            f_pass_weights_random[irx,:] = torch.tensor( [f_pass_weights[irx].item() for irx in random_steps_dx] ).to(device)
  

                    elif args.dynamic_loss_weight_per_fpass_type.startswith("local"):
                        f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)
                


                if args.random_horizon[proto]:
                    horizon_range = torch.arange(1,horizon+1)
                    rand_horizon = horizon_range[torch.randint(horizon_range.size(0), size=(1,))].item()
                else:
                    rand_horizon = horizon


                if args.push_forward[proto]:
                    if args.push_forward_parameter_random[proto]:
                        horizon_grad_range = torch.arange(max(1, rand_horizon-args.push_forward_parameter[proto]),rand_horizon+1)
                        rand_horizon_grad = horizon_grad_range[torch.randint(horizon_grad_range.size(0), size=(1,))].item()
                    else:
                        rand_horizon_grad = max(1, rand_horizon-args.push_forward_parameter[proto])
                else:
                    rand_horizon_grad = rand_horizon



                # import pdb; pdb.set_trace()

                with torch.no_grad():    
                    for t in range(rand_horizon-rand_horizon_grad):
                        #import pdb; pdb.set_trace()

                        y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                        y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                        y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

                        if normalizer:
                            x = normalizer(x)
                        if noise:
                            x = x + torch.randn(x.shape, device=x.device) * noise_std

                        if time_prediction.startswith("constant"):
                            if args.dataset_name.endswith("E1"):
                                out = model(x).to(device)
                            elif args.dataset_name.endswith("E2"):
                                out = model(torch.cat((x, parameters), dim=-1)).to(device)


                        if time_prediction.startswith("variable"):

                            if time_conditioning.startswith("addition"):
                                x_x_t = x + x_t
                                if args.dataset_name.endswith("E1"):
                                    out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                                elif args.dataset_name.endswith("E2"):
                                    out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                            elif time_conditioning.startswith("concatenate"):
                                if args.dataset_name.endswith("E1"):
                                    out = model( x, x_t, y_t ).to(device)
                                elif args.dataset_name.endswith("E2"):
                                    out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                            
                            
                            elif time_conditioning.startswith("attention"):
                                if args.dataset_name.endswith("E1"):
                                    out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                                elif args.dataset_name.endswith("E2"):
                                    out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)

                        
                        train_print_time(args, ep, last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad  )

                        #import pdb; pdb.set_trace()
                        x = torch.cat((x[..., next_input_time_stamps:], out[...,:next_input_time_stamps]), dim=-1)
                        x_t = torch.cat((x_t[..., next_input_time_stamps:], y_t[...,:next_input_time_stamps]), dim=-1)
                        x_tindicies = torch.cat((x_tindicies[..., next_input_time_stamps:], y_tindicies[...,:next_input_time_stamps]), dim=-1)
                        
                        a_l += 1

                        #print("loss->", loss)


                #import pdb; pdb.set_trace()

                for t in range(rand_horizon-rand_horizon_grad, rand_horizon):
                    #import pdb; pdb.set_trace()

                    y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                    y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                    y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]
                

                    if time_prediction.startswith("constant"):
                        if args.dataset_name.endswith("E1"):
                            out = model(x).to(device)
                        elif args.dataset_name.endswith("E2"):
                            out = model(torch.cat((x, parameters), dim=-1)).to(device)

                    #import pdb; pdb.set_trace()
                    if time_prediction.startswith("variable"):

                        if time_conditioning.startswith("addition"):
                            x_x_t = x + x_t
                            if args.dataset_name.endswith("E1"):
                                out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                            elif args.dataset_name.endswith("E2"):
                                out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                        elif time_conditioning.startswith("concatenate"):
                            if args.dataset_name.endswith("E1"):
                                out = model( x, x_t, y_t ).to(device)
                            elif args.dataset_name.endswith("E2"):
                                out = model( torch.cat((x, x_t,y_t, parameters), dim=-1 ) ).to(device)

                        elif time_conditioning.startswith("attention"):
                            if args.dataset_name.endswith("E1"):
                                out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                            elif args.dataset_name.endswith("E2"):
                                out = model(  x, x_tindicies, y_tindicies, parameters).to(device)


                    #import pdb; pdb.set_trace()

                    loss_t = criterion(out, y).sum(dim=[1]).to(device)
                    loss_t = torch.sqrt(loss_t).to(device)

                    loss_t_w = t_step_weights*loss_t

                    loss += (f_pass_weights_random[:,a_l]*loss_t_w.sum(dim=[1])).sum()

                    
                    train_print_time(args, ep,last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad  )
                    #import pdb; pdb.set_trace()

                    x = torch.cat((x[..., next_input_time_stamps:], out[...,:next_input_time_stamps]), dim=-1)
                    x_t = torch.cat((x_t[..., next_input_time_stamps:], y_t[...,:next_input_time_stamps]), dim=-1)
                    x_tindicies = torch.cat((x_tindicies[..., next_input_time_stamps:], y_tindicies[...,:next_input_time_stamps]), dim=-1)
                                        
                    a_l += 1


                train_l2_full += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #import pdb; pdb.set_trace()
        if (count_t_iter) % 500 == 0:
            p.print(f"t_iter: {count_t_iter}/{total_iter}")
            p.print(f"f_pass_weights: {f_pass_weights}")
            p.print(f"f_pass_weights_random: {f_pass_weights_random[:3]}")
            p.print("\n")
            
        
    return train_l2_full, model, count_t_iter






























# def random_time_sampling_new(
#     args,
#     count_t_iter,
#     proto,

#     ep,
#     last_epoch_no,

#     t_iteration,
#     n_tsamples,
#     data_batchsize,

#     model,
#     optimizer,
#     train_loader,
#     criterion,


#     input_time_stamps,
#     output_time_stamps,
#     t_resolution, 

#     timestamps,

#     f_pass_weights,
#     t_step_weights,

#     time_prediction,
#     time_conditioning,

#     horizon,

#     dt_step,

#     noise,
#     noise_std,
#     norm,

#     ):


#     train_l2_full = 0
    

#     #import pdb; pdb.set_trace()

#     total_iter = sum(args.epochs)*t_iteration
#     t_sample_space = torch.arange(t_resolution)

#     if time_prediction == "constant":
#         assert args.time_sampling_choice == 1  # use the right type of random time generator
#         assert dt_step-1 <= int( t_resolution/max(n_tsamples) )

#     # elif time_prediction == "variable":
#     #     assert args.time_sampling_choice > 1

#     for out_samp in range(len(n_tsamples)):
#         assert len(torch.tensor([t for t in range(0, (args.t_resolution -  (out_samp + ((out_samp -1)*(dt_step-1)) ) + 1 ))]) ) > 0 ## check that the length of the initial sample range is positvie 


#     for s in range(t_iteration):
#         #import pdb; pdb.set_trace()
#         count_t_iter += 1

#         for out_samp in range(len(n_tsamples)):

#             tsamples = n_tsamples[out_samp]
#             horizon = args.horizon[out_samp] #(tsamples-output_time_stamps)//output_time_stamps

#             # p.print(f"tsample: {tsamples} ")
#             # p.print(f"horizon: {horizon}")


#             if args.dynamic_loss_weight_per_fpass[proto]:
#                 f_pass_weights = dynamic_weight_loss_sq(count_t_iter, total_iter, args.dynamic_loss_weight_per_fpass_constant_parameter[proto], (args.t_resolution - args.output_time_stamps)//args.output_time_stamps, args.horizon).to(device)
#             elif args.dynamic_loss_weight_per_fpass[proto] == None:
#                 raise TypeError("Specify dynamic_w_l_f_pass (True or False) ")

#             if args.dynamic_loss_weight_per_fpass_reversed[proto]:
#                 f_pass_weights = dynamic_weight_loss_sq(total_iter- count_t_iter + 1, total_iter,args.dynamic_loss_weight_per_fpass_constant_parameter[proto], (args.t_resolution - args.output_time_stamps)//args.output_time_stamps, args.horizon).to(device)
#                 f_pass_weights = torch.flip(f_pass_weights, dims=[0])
#             elif args.dynamic_loss_weight_per_fpass_reversed[proto] == None:
#                 raise TypeError("Specify dynamic_w_l_f_pass_reversed (True or False) ")

#             if args.dynamic_loss_weight_per_tstamp[proto]:
#                 t_step_weights = dynamic_weight_loss(count_t_iter, total_iter, args.dynamic_loss_weight_per_tstamp_constant_parameter[proto],  args.output_time_stamps  ).to(device)
#             elif args.dynamic_loss_weight_per_tstamp[proto] == None:
#                 raise TypeError("Specify dynamic_w_l_t_steps (True or False) ")


#             #import pdb; pdb.set_trace()

#             for (data, u_super, x, parameters) in train_loader:

#                 #import pdb; pdb.set_trace()
#                 data = data.to(device)
#                 parameters = parameters[...,:args.no_parameters].to(device)

#                 time_sampling_choice = args.time_sampling_choice
#                 data_batch = batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
#                 time_indicies = t_sample_space[data_batch.indicies]
#                 xy = torch.gather(data, -1, time_indicies.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
#                 xy_t = torch.ones_like(xy)[:,0,:].to(device)
#                 xy_t = xy_t*timestamps[time_indicies]
#                 xy_t = xy_t.unsqueeze(1).repeat(1,data.shape[1],1)
#                 xy_tindicies = time_indicies.long()

#                 #time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
#                 time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
        
#                 x = xy[..., :input_time_stamps ]
#                 x_t = xy_t[..., :input_time_stamps ]
#                 x_tindicies = xy_tindicies[..., :input_time_stamps ]

#                 # if s < 5:
#                 #     print("time_stamps->", time_stamps)
#                 #     print("starting_index->", x_tindicies)


                
#                 loss = 0
#                 a_l = 0


#                 #import pdb; pdb.set_trace()
#                 f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)

#                 if args.dynamic_loss_weight_per_fpass[proto]:
                    
#                     # if args.dynamic_loss_weight_per_fpass_type.startswith("global"):
#                     #     random_steps = time_indicies[:,output_time_stamps]
#                     #     f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
#                     #     random_steps_dx = [ torch.div(irx, output_time_stamps, rounding_mode='floor')-1 for irx in random_steps]
#                     #     for irx in range(len(random_steps_dx)):
#                     #         f_pass_weights_random[irx,:] = torch.tensor([f_pass_weights[irx].item() for irx in range(random_steps_dx[irx],random_steps_dx[irx]+horizon)]).to(device)

#                     if args.dynamic_loss_weight_per_fpass_type.startswith("global"):
#                         random_steps = time_indicies[:,output_time_stamps:][:,::output_time_stamps]
#                         f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
#                         for irx in range(args.batch_size_train):
#                             random_steps_dx = [ torch.div(irx-output_time_stamps, output_time_stamps, rounding_mode='floor').item() for irx in random_steps[irx]]
#                             f_pass_weights_random[irx,:] = torch.tensor( [f_pass_weights[irx].item() for irx in random_steps_dx] ).to(device)
  

#                     elif args.dynamic_loss_weight_per_fpass_type.startswith("local"):
#                         f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)
                


#                 if args.random_horizon[proto]:
#                     horizon_range = torch.arange(1,horizon+1)
#                     rand_horizon = horizon_range[torch.randint(horizon_range.size(0), size=(1,))].item()
#                 else:
#                     rand_horizon = horizon


#                 if args.push_forward[proto]:
#                     if args.push_forward_parameter_random[proto]:
#                         horizon_grad_range = torch.arange(max(1, rand_horizon-args.push_forward_parameter[proto]),rand_horizon+1)
#                         rand_horizon_grad = horizon_grad_range[torch.randint(horizon_grad_range.size(0), size=(1,))].item()
#                     else:
#                         rand_horizon_grad = max(1, rand_horizon-args.push_forward_parameter[proto])
#                 else:
#                     rand_horizon_grad = rand_horizon



#                 # import pdb; pdb.set_trace()

#                 with torch.no_grad():    
#                     for t in range(rand_horizon-rand_horizon_grad):
#                         #import pdb; pdb.set_trace()

#                         # y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
#                         # y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
#                         # y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

#                         y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
#                         y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
#                         y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]


#                         if time_prediction.startswith("constant"):
#                             if args.dataset_name.endswith("E1"):
#                                 out = model(x).to(device)
#                             elif args.dataset_name.endswith("E2"):
#                                 out = model(torch.cat((x, parameters), dim=-1)).to(device)


#                         if time_prediction.startswith("variable"):

#                             if time_conditioning.startswith("addition"):
#                                 x_x_t = x + x_t
#                                 if args.dataset_name.endswith("E1"):
#                                     out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
#                                 elif args.dataset_name.endswith("E2"):
#                                     out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

#                             elif time_conditioning.startswith("concatenate"):
#                                 if args.dataset_name.endswith("E1"):
#                                     out = model( x, x_t, y_t ).to(device)
#                                 elif args.dataset_name.endswith("E2"):
#                                     out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                            
                            
#                             elif time_conditioning.startswith("attention"):
#                                 if args.dataset_name.endswith("E1"):
#                                     out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
#                                 elif args.dataset_name.endswith("E2"):
#                                     out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


#                         train_print_time(args, ep, last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )

#                         #import pdb; pdb.set_trace()
#                         if output_time_stamps > input_time_stamps:
#                             x = out[...,-input_time_stamps:]
#                             x_t = y_t[...,-input_time_stamps:]
#                             x_tindicies = y_tindicies[...,-input_time_stamps:]

#                         elif output_time_stamps == input_time_stamps:
#                             x = torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
#                             x_t = torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
#                             x_tindicies = torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                        
#                         elif output_time_stamps < input_time_stamps:
#                             x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
#                             x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
#                             x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                        
#                         a_l += 1

#                         #print("loss->", loss)


#                 #import pdb; pdb.set_trace()

#                 for t in range(rand_horizon-rand_horizon_grad, rand_horizon):
#                     #import pdb; pdb.set_trace()

#                     y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
#                     y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
#                     y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                

#                     if norm:
#                         x = normalizer(x)
#                     if noise:
#                         x = x + torch.randn(x.shape, device=x.device) * noise_std

#                     if time_prediction.startswith("constant"):
#                         if args.dataset_name.endswith("E1"):
#                             out = model(x).to(device)
#                         elif args.dataset_name.endswith("E2"):
#                             out = model(torch.cat((x, parameters), dim=-1)).to(device)

#                     #import pdb; pdb.set_trace()
#                     if time_prediction.startswith("variable"):

#                         if time_conditioning.startswith("addition"):
#                             x_x_t = x + x_t
#                             if args.dataset_name.endswith("E1"):
#                                 out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
#                             elif args.dataset_name.endswith("E2"):
#                                 out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

#                         elif time_conditioning.startswith("concatenate"):
#                             if args.dataset_name.endswith("E1"):
#                                 out = model( x, x_t, y_t ).to(device)
#                             elif args.dataset_name.endswith("E2"):
#                                 out = model( torch.cat((x, x_t,y_t, parameters), dim=-1 ) ).to(device)

#                         elif time_conditioning.startswith("attention"):
#                             if args.dataset_name.endswith("E1"):
#                                 out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
#                             elif args.dataset_name.endswith("E2"):
#                                 out = model(  x, x_tindicies, y_tindicies, parameters).to(device)

#                     if norm:
#                         out = normalizer.inverse(out)


#                     #import pdb; pdb.set_trace()

#                     loss_t = criterion(out, y).sum(dim=[1]).to(device)
#                     loss_t = torch.sqrt(loss_t).to(device)

#                     loss_t_w = t_step_weights*loss_t

#                     loss += (f_pass_weights_random[:,a_l]*loss_t_w.sum(dim=[1])).sum()

                    
#                     train_print_time(args, ep,last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )
#                     #import pdb; pdb.set_trace()

#                     if output_time_stamps > input_time_stamps:
#                         x = out[...,-input_time_stamps:]
#                         x_t = y_t[...,-input_time_stamps:]
#                         x_tindicies = y_tindicies[...,-input_time_stamps:]

#                     elif output_time_stamps == input_time_stamps:
#                         x = torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
#                         x_t = torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
#                         x_tindicies = torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                    
#                     elif output_time_stamps < input_time_stamps:
#                         x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
#                         x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
#                         x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                    
#                     a_l += 1


#                 train_l2_full += loss.item()

#                 # Backward pass
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         #import pdb; pdb.set_trace()
#         if (count_t_iter) % 500 == 0:
#             p.print(f"t_iter: {count_t_iter}/{total_iter}")
#             p.print(f"f_pass_weights: {f_pass_weights}")
#             p.print(f"f_pass_weights_random: {f_pass_weights_random[:3]}")
#             p.print("\n")
            
        
#     return train_l2_full, model, count_t_iter
























def random_time_sampling_new(
    args,
    count_t_iter,
    proto,

    ep,
    epochs,
    last_epoch_no,

    t_iteration,
    n_tsamples,
    data_batchsize,

    model,
    optimizer,
    train_loader,
    criterion,


    input_time_stamps,
    output_time_stamps,
    t_resolution, 

    timestamps,

    f_pass_weights,
    t_step_weights,

    time_prediction,
    time_conditioning,

    max_horizon,
    horizons,
    random_horizon,
    

    dt_step,

    noise,
    noise_std,
    norm,

    scheduler,
    sheduler_change,


    dynamic_loss_weight_per_fpass,
    dynamic_loss_weight_per_fpass_type,
    dynamic_loss_weight_per_fpass_reversed,
    dynamic_loss_weight_per_fpass_constant_parameter,

    dynamic_loss_weight_per_tstamp,
    dynamic_loss_weight_per_tstamp_constant_parameter,

    push_forward,
    push_forward_parameter_random,
    push_forward_parameter,

    ):


    train_l2_full = 0
    

    #import pdb; pdb.set_trace()

    total_iter = epochs*t_iteration
    t_sample_space = torch.arange(t_resolution)

    # if args.time_prediction == "constant":
    #     assert args.time_sampling_choice == 1  # use the right type of random time generator
    #     assert dt_step-1 <= int( t_resolution/max(n_tsamples) )

    # elif args.time_prediction == "variable":
    #     assert args.time_sampling_choice > 1


    for out_samp in range(len(n_tsamples)):
        assert len(torch.tensor([t for t in range(0, (t_resolution -  (out_samp + ((out_samp -1)*(dt_step-1)) ) + 1 ))]) ) > 0 ## check that the length of the initial sample range is positvie 

    #max_horizon = round(( t_resolution - input_time_stamps)//output_time_stamps)


    
    for s in range(t_iteration):
        #import pdb; pdb.set_trace()
        count_t_iter += 1

        for out_samp in range(len(n_tsamples)):
            #import pdb; pdb.set_trace()

            tsamples = n_tsamples[out_samp]
            horizon = horizons[out_samp] #(tsamples-output_time_stamps)//output_time_stamps

            # p.print(f"tsample: {tsamples} ")
            # p.print(f"horizon: {horizon}")
            

            if dynamic_loss_weight_per_fpass:
                f_pass_weights = dynamic_weight_loss_sq(count_t_iter, total_iter, dynamic_loss_weight_per_fpass_constant_parameter, max_horizon, horizon).to(device)
                #f_pass_weights = torch.flip(f_pass_weights, dims=[0])
            elif dynamic_loss_weight_per_fpass == None:
                raise TypeError("Specify dynamic_w_l_f_pass (True or False) ")
            

            # ww = dynamic_weight_loss_sq(count_t_iter, total_iter, dynamic_loss_weight_per_fpass_constant_parameter, max_horizon, horizon).to(device)
            # ww_reversed = dynamic_weight_loss_sq(total_iter- count_t_iter + 1, total_iter,dynamic_loss_weight_per_fpass_constant_parameter, max_horizon, horizon).to(device)
            # p.print(f"weights: {ww}")
            # p.print(f"weights_flip: {torch.flip(ww, dims=[0])}")
            # p.print(f"weights_reversed: {ww}")
            # p.print(f"weights_reversed_flip: {torch.flip(ww_reversed, dims=[0])}")


            if dynamic_loss_weight_per_fpass_reversed:
                f_pass_weights = dynamic_weight_loss_sq(total_iter- count_t_iter + 1, total_iter,dynamic_loss_weight_per_fpass_constant_parameter, max_horizon, horizon).to(device)
                f_pass_weights = torch.flip(f_pass_weights, dims=[0])
            elif dynamic_loss_weight_per_fpass_reversed == None:
                raise TypeError("Specify dynamic_w_l_f_pass_reversed (True or False) ")

            if dynamic_loss_weight_per_tstamp:
                t_step_weights = dynamic_weight_loss(count_t_iter, total_iter, dynamic_loss_weight_per_tstamp_constant_parameter,  output_time_stamps  ).to(device)
            elif dynamic_loss_weight_per_tstamp == None:
                raise TypeError("Specify dynamic_w_l_t_steps (True or False) ")


            #import pdb; pdb.set_trace()

            for bb, (data, u_super, x, parameters) in enumerate(train_loader):
                #import pdb; pdb.set_trace()

                #import pdb; pdb.set_trace()
                data = data.to(device)
                parameters = parameters[...,:args.no_parameters].to(device)

                time_sampling_choice = args.time_sampling_choice
                data_batch = batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
                
                time_indicies = t_sample_space[data_batch.indicies]


                xy = torch.gather(data, -1, time_indicies.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
                xy_t = torch.ones_like(xy)[:,0,:].to(device)
                xy_t = xy_t*timestamps[time_indicies]


                # p.print(f"xy_t: {xy_t.shape}")
                # p.print(f"xy_t: {xy_t[:3,:10]}")
                xy_t = torch.cat((torch.diff(xy_t, dim=-1), torch.zeros(xy_t.shape[0], 1).to(device)), dim=-1)
                # p.print(f"xy_t: {xy_t.shape}")
                # p.print(f"xy_t: {xy_t[:3,:10]}")
                

                xy_t = xy_t.unsqueeze(1).repeat(1,data.shape[1],1)
                xy_tindicies = time_indicies.long()

                #time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
                time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
        
                x = xy[..., :input_time_stamps ]
                x_t = xy_t[..., :input_time_stamps ]
                x_tindicies = xy_tindicies[..., :input_time_stamps ]

                # if s < 5:
                #     print("time_stamps->", time_stamps)
                #     print("starting_index->", x_tindicies)


                
                loss = 0
                a_l = 0


                #import pdb; pdb.set_trace()
                f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)

                if dynamic_loss_weight_per_fpass:
                    
                    # if args.dynamic_loss_weight_per_fpass_type.startswith("global"):
                    #     random_steps = time_indicies[:,output_time_stamps]
                    #     f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
                    #     random_steps_dx = [ torch.div(irx, output_time_stamps, rounding_mode='floor')-1 for irx in random_steps]
                    #     for irx in range(len(random_steps_dx)):
                    #         f_pass_weights_random[irx,:] = torch.tensor([f_pass_weights[irx].item() for irx in range(random_steps_dx[irx],random_steps_dx[irx]+horizon)]).to(device)

                    if dynamic_loss_weight_per_fpass_type.startswith("global"):
                        random_steps = time_indicies[:,output_time_stamps:][:,::output_time_stamps]
                        f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
                        for irx in range(args.batch_size_train):
                            random_steps_dx = [ torch.div(irx-output_time_stamps, output_time_stamps, rounding_mode='floor').item() for irx in random_steps[irx]]
                            f_pass_weights_random[irx,:] = torch.tensor( [f_pass_weights[irx].item() for irx in random_steps_dx] ).to(device)
  

                    elif dynamic_loss_weight_per_fpass_type.startswith("local"):
                        f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)
                


                if random_horizon:
                    horizon_range = torch.arange(1,horizon+1)
                    rand_horizon = horizon_range[torch.randint(horizon_range.size(0), size=(1,))].item()
                else:
                    rand_horizon = horizon


                if push_forward:
                    if push_forward_parameter_random:
                        horizon_grad_range = torch.arange(max(1, rand_horizon-push_forward_parameter),rand_horizon+1)
                        rand_horizon_grad = horizon_grad_range[torch.randint(horizon_grad_range.size(0), size=(1,))].item()
                    else:
                        rand_horizon_grad = max(1, rand_horizon-push_forward_parameter)
                else:
                    rand_horizon_grad = rand_horizon



                # import pdb; pdb.set_trace()

                with torch.no_grad():    
                    for t in range(rand_horizon-rand_horizon_grad):
                        #import pdb; pdb.set_trace()
                        #import pdb; pdb.set_trace()

                        # y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                        # y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                        # y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

                        y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                        y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                        y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]


                        if time_prediction == "constant":
                            if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                                out = model(x).to(device)
                            elif args.dataset_name == "E2":
                                out = model(torch.cat((x, parameters), dim=-1)).to(device)
                            elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
                                out = model(torch.cat((x, parameters), dim=-1)).to(device)


                        if time_prediction == "variable":

                            if time_conditioning == "addition":
                                x_x_t = x + x_t
                                if args.dataset_name == "E1" or "B1" or "A1":
                                    out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                                elif args.dataset_name == "E2":
                                    out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                            elif time_conditioning == "concatenate":
                                if args.dataset_name == "E1" or "B1"  or "A1":
                                    out = model( x, x_t, y_t ).to(device)
                                elif args.dataset_name == "E2":
                                    out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                            
                            
                            elif time_conditioning == "attention":
                                if args.dataset_name == "E1" or "B1"  or "A1":
                                    out = model( x.to(device), x_t.to(device), y_t.to(device) ).to(device)
                                elif args.dataset_name == "E2":
                                    out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


                        train_print_time(args, ep, last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )

                        #import pdb; pdb.set_trace()


                        if output_time_stamps > input_time_stamps:
                            x = out[...,-input_time_stamps:]
                            x_t = y_t[...,-input_time_stamps:]
                            x_tindicies = y_tindicies[...,-input_time_stamps:]

                        elif output_time_stamps == input_time_stamps:
                            x = torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                            x_t = torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                            x_tindicies = torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                        
                        elif output_time_stamps < input_time_stamps:
                            x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
                            x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                            x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                        
                        a_l += 1

                        #print("loss->", loss)


                #import pdb; pdb.set_trace()
                for t in range(rand_horizon-rand_horizon_grad, rand_horizon):
                    #import pdb; pdb.set_trace()
                    #import pdb; pdb.set_trace()

                    y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                    y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                    y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                

                    # if noise:
                    #     if norm:
                    #         x = normalizer(x)
                    #         x = x + torch.randn(x.shape, device=x.device) * noise_std
                    #     x = x  + torch.randn(x.shape, device=x.device) * torch.max(x)*noise_std


                    if norm:
                        x = normalizer(x)

                    if noise:
                        x = x  + torch.randn(x.shape, device=x.device) * torch.std(x)*noise_std

                    if time_prediction == "constant":
                        if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                            out = model(x).to(device)
                        elif args.dataset_name == "E2":
                            out = model(torch.cat((x, parameters), dim=-1)).to(device)
                        elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
                            out = model(torch.cat((x, parameters), dim=-1)).to(device)


                    #import pdb; pdb.set_trace()
                    if time_prediction == "variable":

                        if time_conditioning == "addition":
                            x_x_t = x + x_t
                            if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                                out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                            elif args.dataset_name == "E2":
                                out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                        elif time_conditioning == "concatenate":
                            if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                                out = model( x, x_t, y_t ).to(device)
                            elif args.dataset_name == "E2":
                                out = model( torch.cat((x, x_t, y_t, parameters), dim=-1 ) ).to(device)

                        elif time_conditioning == "attention":
                            if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                                #x.permute(0,2,1).to(device), y.permute(0,2,1).to(device),x_mask, y_mask, x_t[:, 0, :].to(device), y_t[:, 0, :].to(device)
                                #print("x, x_t, y_t -->", x.shape, x_t.shape, y_t.shape)
                                out = model( x, x_t, y_t ).to(device)
                                #print("y, out", y.shape, out.shape)

                                #out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                            elif args.dataset_name == "E2":
                                out = model(  x, x_tindicies, y_tindicies, parameters).to(device)
                    
                                
                    if norm:
                        out = normalizer.inverse(out)


                    #args.predict_difference = True
                    if args.predict_difference:
                        if args.dataset_name == "KS1":
                            out = x + 0.3*out
                        else:
                            out = x + out

                    #import pdb; pdb.set_trace()


                    #p.print(f"out, y: {out.shape}, {y.shape}")

                    
                    loss_t = criterion(out, y).to(device)                #### FOR L2
                    #loss_t = criterion(out, y).sum(dim=[1]).to(device)     ########### FOR MSE


                    

                    #p.print(f"loss_t: {loss_t}")
                    #p.print(f"loss_t: {loss_t.shape}")
                    #p.print(f"criterion: {criterion}")
                    # loss += loss_t



                    #loss_t = criterion(out, y).sum(dim=[1]).to(device)

                    loss_t = torch.sqrt(loss_t).to(device)

                    #p.print(f"loss_t, t_step_weights:  {loss_t.shape}, {t_step_weights.shape}")
                    loss_t_w = t_step_weights*loss_t

                    #p.print(f"loss_t_w, t_step_weights:  {loss_t_w.shape}, {f_pass_weights_random[:,a_l].shape}")
                    loss += (f_pass_weights_random[:,a_l]*loss_t_w.sum(dim=[1])).sum()

                    

                    #print("\n")
                    train_print_time(args, ep,last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )
                    #import pdb; pdb.set_trace()



                    # curiculum_learning = False
                    # #print("ep, epochs, k k_transtion -->",ep, args.epochs[proto], t, k_transition(ep, args.t_resolution, args.epochs[proto]) ) 
                    # if curiculum_learning and t >= k_transition(ep, t_resolution, epochs):
                        
                    #     # if s % 500 == 0:
                    #     #     print("ep, epochs, k k_transtion -->",ep, args.epochs[proto], t, k_transition(ep, args.t_resolution, args.epochs[proto]) )
                    #         #print("switching to true solution")
                    #     out = y
                    

                        
                    if output_time_stamps > input_time_stamps:
                        x = out[...,-input_time_stamps:]
                        x_t = y_t[...,-input_time_stamps:]
                        x_tindicies = y_tindicies[...,-input_time_stamps:]

                    elif output_time_stamps == input_time_stamps:
                        x = torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                        x_t = torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                        x_tindicies = torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                    
                    elif output_time_stamps < input_time_stamps:
                        x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
                        x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                        x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                    
                    a_l += 1


                train_l2_full += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        if sheduler_change == "iteration":
            #p.print(f"learning_rate: { optimizer.param_groups[0]['lr']}" )
            scheduler.step()


        #import pdb; pdb.set_trace()
        if (count_t_iter) % 500 == 0:
            p.print(f"t_iter: {count_t_iter}/{total_iter}")
            p.print(f"f_pass_weights: {f_pass_weights}")
            p.print(f"f_pass_weights_random: {f_pass_weights_random[:3]}")
            p.print("\n")
            
        
    return train_l2_full/(t_iteration*(bb+1)), model, count_t_iter
















def random_time_sampling_new_new(
    args,
    count_t_iter,
    proto,

    ep,
    epochs,
    last_epoch_no,

    t_iteration,
    n_tsamples,
    data_batchsize,

    model,
    optimizer,
    train_loader,
    criterion,


    input_time_stamps,
    output_time_stamps,
    t_resolution, 

    timestamps,

    f_pass_weights,
    t_step_weights,

    time_prediction,
    time_conditioning,

    max_horizon,
    horizons,
    random_horizon,
    

    dt_step,

    noise,
    noise_std,
    norm,

    scheduler,
    sheduler_change,


    dynamic_loss_weight_per_fpass,
    dynamic_loss_weight_per_fpass_type,
    dynamic_loss_weight_per_fpass_reversed,
    dynamic_loss_weight_per_fpass_constant_parameter,

    dynamic_loss_weight_per_tstamp,
    dynamic_loss_weight_per_tstamp_constant_parameter,

    push_forward,
    push_forward_parameter_random,
    push_forward_parameter,

    ):


    train_l2_full = 0
    

    #import pdb; pdb.set_trace()

    total_iter = epochs*t_iteration
    t_sample_space = torch.arange(t_resolution)

    if time_prediction == "constant":
        assert args.time_sampling_choice == 1  # use the right type of random time generator
        assert dt_step-1 <= int( t_resolution/max(n_tsamples) )

    # elif time_prediction == "variable":
    #     assert args.time_sampling_choice > 1

    for out_samp in range(len(n_tsamples)):
        assert len(torch.tensor([t for t in range(0, (t_resolution -  (out_samp + ((out_samp -1)*(dt_step-1)) ) + 1 ))]) ) > 0 ## check that the length of the initial sample range is positvie 

    #max_horizon = round(( t_resolution - input_time_stamps)//output_time_stamps)


    
    # for s in range(t_iteration):
    #     #import pdb; pdb.set_trace()
    #     count_t_iter += 1

    #     for out_samp in range(len(n_tsamples)):
    #         #import pdb; pdb.set_trace()

    #         tsamples = n_tsamples[out_samp]
    #         horizon = horizons[out_samp] #(tsamples-output_time_stamps)//output_time_stamps

    #         # p.print(f"tsample: {tsamples} ")
    #         # p.print(f"horizon: {horizon}")
            



    for out_samp in range(len(n_tsamples)):
        #import pdb; pdb.set_trace()

        tsamples = n_tsamples[out_samp]
        horizon = horizons[out_samp] #(tsamples-output_time_stamps)//output_time_stamps


        time_sampling_choice = args.time_sampling_choice
        data_batch = batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(t_iteration*data_batchsize, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
        #p.print(f"data_batch.indicies: {data_batch.indicies.shape}")
        data_batch.indicies = data_batch.indicies.reshape(t_iteration, data_batchsize, tsamples)
        #p.print(f"data_batch.indicies: {data_batch.indicies.shape}")


        for s in range(t_iteration):
        #import pdb; pdb.set_trace()
            count_t_iter += 1

            # p.print(f"tsample: {tsamples} ")
            # p.print(f"horizon: {horizon}")

            if dynamic_loss_weight_per_fpass:
                f_pass_weights = dynamic_weight_loss_sq(count_t_iter, total_iter, dynamic_loss_weight_per_fpass_constant_parameter, max_horizon, horizon).to(device)
            elif dynamic_loss_weight_per_fpass == None:
                raise TypeError("Specify dynamic_w_l_f_pass (True or False) ")

            if dynamic_loss_weight_per_fpass_reversed:
                f_pass_weights = dynamic_weight_loss_sq(total_iter- count_t_iter + 1, total_iter,dynamic_loss_weight_per_fpass_constant_parameter, max_horizon, horizon).to(device)
                f_pass_weights = torch.flip(f_pass_weights, dims=[0])
            elif dynamic_loss_weight_per_fpass_reversed == None:
                raise TypeError("Specify dynamic_w_l_f_pass_reversed (True or False) ")

            if dynamic_loss_weight_per_tstamp:
                t_step_weights = dynamic_weight_loss(count_t_iter, total_iter, dynamic_loss_weight_per_tstamp_constant_parameter,  output_time_stamps  ).to(device)
            elif dynamic_loss_weight_per_tstamp == None:
                raise TypeError("Specify dynamic_w_l_t_steps (True or False) ")


            #import pdb; pdb.set_trace()

            for bb, (data, u_super, x, parameters) in enumerate(train_loader):
                #import pdb; pdb.set_trace()

                #import pdb; pdb.set_trace()
                data = data.to(device)
                parameters = parameters[...,:args.no_parameters].to(device)

                # time_sampling_choice = args.time_sampling_choice
                # data_batch = batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
                
                time_indicies = t_sample_space[data_batch.indicies[s]]


                xy = torch.gather(data, -1, time_indicies.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
                xy_t = torch.ones_like(xy)[:,0,:].to(device)
                xy_t = xy_t*timestamps[time_indicies]
                xy_t = xy_t.unsqueeze(1).repeat(1,data.shape[1],1)
                xy_tindicies = time_indicies.long()

                #time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
                time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
        
                x = xy[..., :input_time_stamps ]
                x_t = xy_t[..., :input_time_stamps ]
                x_tindicies = xy_tindicies[..., :input_time_stamps ]

                # if s < 5:
                #     print("time_stamps->", time_stamps)
                #     print("starting_index->", x_tindicies)


                
                loss = 0
                a_l = 0


                #import pdb; pdb.set_trace()
                f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)

                if dynamic_loss_weight_per_fpass:
                    
                    # if args.dynamic_loss_weight_per_fpass_type.startswith("global"):
                    #     random_steps = time_indicies[:,output_time_stamps]
                    #     f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
                    #     random_steps_dx = [ torch.div(irx, output_time_stamps, rounding_mode='floor')-1 for irx in random_steps]
                    #     for irx in range(len(random_steps_dx)):
                    #         f_pass_weights_random[irx,:] = torch.tensor([f_pass_weights[irx].item() for irx in range(random_steps_dx[irx],random_steps_dx[irx]+horizon)]).to(device)

                    if dynamic_loss_weight_per_fpass_type.startswith("global"):
                        random_steps = time_indicies[:,output_time_stamps:][:,::output_time_stamps]
                        f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
                        for irx in range(args.batch_size_train):
                            random_steps_dx = [ torch.div(irx-output_time_stamps, output_time_stamps, rounding_mode='floor').item() for irx in random_steps[irx]]
                            f_pass_weights_random[irx,:] = torch.tensor( [f_pass_weights[irx].item() for irx in random_steps_dx] ).to(device)
  

                    elif dynamic_loss_weight_per_fpass_type.startswith("local"):
                        f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)
                


                if random_horizon:
                    horizon_range = torch.arange(1,horizon+1)
                    rand_horizon = horizon_range[torch.randint(horizon_range.size(0), size=(1,))].item()
                else:
                    rand_horizon = horizon


                if push_forward:
                    if push_forward_parameter_random:
                        horizon_grad_range = torch.arange(max(1, rand_horizon-push_forward_parameter),rand_horizon+1)
                        rand_horizon_grad = horizon_grad_range[torch.randint(horizon_grad_range.size(0), size=(1,))].item()
                    else:
                        rand_horizon_grad = max(1, rand_horizon-push_forward_parameter)
                else:
                    rand_horizon_grad = rand_horizon



                # import pdb; pdb.set_trace()

                with torch.no_grad():    
                    for t in range(rand_horizon-rand_horizon_grad):
                        #import pdb; pdb.set_trace()
                        #import pdb; pdb.set_trace()

                        # y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                        # y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                        # y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

                        y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                        y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                        y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]


                        if time_prediction == "constant":
                            if args.dataset_name == "E1":
                                out = model(x).to(device)
                            elif args.dataset_name == "E2":
                                out = model(torch.cat((x, parameters), dim=-1)).to(device)
                            elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
                                out = model(torch.cat((x, parameters), dim=-1)).to(device)


                        if time_prediction == "variable":

                            if time_conditioning == "addition":
                                x_x_t = x + x_t
                                if args.dataset_name == "E1" or "B1":
                                    out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                                elif args.dataset_name == "E2":
                                    out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                            elif time_conditioning == "concatenate":
                                if args.dataset_name == "E1" or "B1":
                                    out = model( x, x_t, y_t ).to(device)
                                elif args.dataset_name == "E2":
                                    out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                            
                            
                            elif time_conditioning == "attention":
                                if args.dataset_name == "E1" or "B1":
                                    out = model( x.to(device), x_t.to(device), y_t.to(device) ).to(device)
                                elif args.dataset_name == "E2":
                                    out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


                        train_print_time(args, ep, last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )

                        #import pdb; pdb.set_trace()


                        if output_time_stamps > input_time_stamps:
                            x = out[...,-input_time_stamps:]
                            x_t = y_t[...,-input_time_stamps:]
                            x_tindicies = y_tindicies[...,-input_time_stamps:]

                        elif output_time_stamps == input_time_stamps:
                            x = torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                            x_t = torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                            x_tindicies = torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                        
                        elif output_time_stamps < input_time_stamps:
                            x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
                            x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                            x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                        
                        a_l += 1

                        #print("loss->", loss)


                #import pdb; pdb.set_trace()
                for t in range(rand_horizon-rand_horizon_grad, rand_horizon):
                    #import pdb; pdb.set_trace()
                    #import pdb; pdb.set_trace()

                    y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                    y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                    y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                

                    # if noise:
                    #     if norm:
                    #         x = normalizer(x)
                    #         x = x + torch.randn(x.shape, device=x.device) * noise_std
                    #     x = x  + torch.randn(x.shape, device=x.device) * torch.max(x)*noise_std


                    if norm:
                        x = normalizer(x)

                    if noise:
                        x = x  + torch.randn(x.shape, device=x.device) * torch.std(x)*noise_std


                    if time_prediction == "constant":
                        if args.dataset_name == "E1" or "B1":
                            out = model(x).to(device)
                        elif args.dataset_name == "E2":
                            out = model(torch.cat((x, parameters), dim=-1)).to(device)
                        elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
                            out = model(torch.cat((x, parameters), dim=-1)).to(device)


                    #import pdb; pdb.set_trace()
                    if time_prediction == "variable":

                        if time_conditioning == "addition":
                            x_x_t = x + x_t
                            if args.dataset_name == "E1" or "B1":
                                out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                            elif args.dataset_name == "E2":
                                out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                        elif time_conditioning == "concatenate":
                            if args.dataset_name == "E1" or "B1":
                                out = model( x, x_t, y_t ).to(device)
                            elif args.dataset_name == "E2":
                                out = model( torch.cat((x, x_t, y_t, parameters), dim=-1 ) ).to(device)

                        elif time_conditioning == "attention":
                            if args.dataset_name == "E1" or "B1":
                                #x.permute(0,2,1).to(device), y.permute(0,2,1).to(device),x_mask, y_mask, x_t[:, 0, :].to(device), y_t[:, 0, :].to(device)
                                #print("x, x_t, y_t -->", x.shape, x_t.shape, y_t.shape)
                                out = model( x, x_t, y_t ).to(device)
                                #print("y, out", y.shape, out.shape)

                                #out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                            elif args.dataset_name == "E2":
                                out = model(  x, x_tindicies, y_tindicies, parameters).to(device)
                    
                                
                    if norm:
                        out = normalizer.inverse(out)


                    #args.predict_difference = True
                    if args.predict_difference:
                        if args.dataset_name == "KS1":
                            out = x + 0.3*out
                        else:
                            out = x + out

                    #import pdb; pdb.set_trace()


                    #p.print(f"out, y: {out.shape}, {y.shape}")

                    
                    loss_t = criterion(out, y).to(device)                #### FOR L2
                    #loss_t = criterion(out, y).sum(dim=[1]).to(device)     ########### FOR MSE


                    

                    #p.print(f"loss_t: {loss_t}")
                    #p.print(f"loss_t: {loss_t.shape}")
                    #p.print(f"criterion: {criterion}")
                    # loss += loss_t



                    #loss_t = criterion(out, y).sum(dim=[1]).to(device)

                    loss_t = torch.sqrt(loss_t).to(device)

                    #p.print(f"loss_t, t_step_weights:  {loss_t.shape}, {t_step_weights.shape}")
                    loss_t_w = t_step_weights*loss_t

                    #p.print(f"loss_t_w, t_step_weights:  {loss_t_w.shape}, {f_pass_weights_random[:,a_l].shape}")
                    loss += (f_pass_weights_random[:,a_l]*loss_t_w.sum(dim=[1])).sum()

                    

                    #print("\n")
                    train_print_time(args, ep,last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )
                    #import pdb; pdb.set_trace()



                    # curiculum_learning = False
                    # #print("ep, epochs, k k_transtion -->",ep, args.epochs[proto], t, k_transition(ep, args.t_resolution, args.epochs[proto]) ) 
                    # if curiculum_learning and t >= k_transition(ep, t_resolution, epochs):
                        
                    #     # if s % 500 == 0:
                    #     #     print("ep, epochs, k k_transtion -->",ep, args.epochs[proto], t, k_transition(ep, args.t_resolution, args.epochs[proto]) )
                    #         #print("switching to true solution")
                    #     out = y
                    

                        
                    if output_time_stamps > input_time_stamps:
                        x = out[...,-input_time_stamps:]
                        x_t = y_t[...,-input_time_stamps:]
                        x_tindicies = y_tindicies[...,-input_time_stamps:]

                    elif output_time_stamps == input_time_stamps:
                        x = torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                        x_t = torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                        x_tindicies = torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                    
                    elif output_time_stamps < input_time_stamps:
                        x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
                        x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                        x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                    
                    a_l += 1


                train_l2_full += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        if sheduler_change == "iteration":
            #p.print(f"learning_rate: { optimizer.param_groups[0]['lr']}" )
            scheduler.step()


        #import pdb; pdb.set_trace()
        if (count_t_iter) % 500 == 0:
            p.print(f"t_iter: {count_t_iter}/{total_iter}")
            p.print(f"f_pass_weights: {f_pass_weights}")
            p.print(f"f_pass_weights_random: {f_pass_weights_random[:3]}")
            p.print("\n")
            
        
    return train_l2_full/(t_iteration*(bb+1)), model, count_t_iter















































def random_time_sampling_one_sample(
    args,
    count_t_iter,
    proto,

    ep,
    epochs,
    last_epoch_no,

    t_iteration,
    n_tsamples,
    data_batchsize,

    model,
    optimizer,
    train_loader,
    criterion,


    input_time_stamps,
    output_time_stamps,
    t_resolution, 

    timestamps,

    f_pass_weights,
    t_step_weights,

    time_prediction,
    time_conditioning,

    max_horizon,
    horizons,
    random_horizon,
    

    dt_step,

    noise,
    noise_std,
    norm,

    scheduler,
    sheduler_change,


    dynamic_loss_weight_per_fpass,
    dynamic_loss_weight_per_fpass_type,
    dynamic_loss_weight_per_fpass_reversed,
    dynamic_loss_weight_per_fpass_constant_parameter,

    dynamic_loss_weight_per_tstamp,
    dynamic_loss_weight_per_tstamp_constant_parameter,

    push_forward,
    push_forward_parameter_random,
    push_forward_parameter,

    ):


    train_l2_full = 0
    

    #import pdb; pdb.set_trace()

    total_iter = epochs*t_iteration
    t_sample_space = torch.arange(t_resolution)

    if time_prediction == "constant":
        assert args.time_sampling_choice == 1  # use the right type of random time generator
        assert dt_step-1 <= int( t_resolution/max(n_tsamples) )

    # elif time_prediction == "variable":
    #     assert args.time_sampling_choice > 1

    for out_samp in range(len(n_tsamples)):
        assert len(torch.tensor([t for t in range(0, (t_resolution -  (out_samp + ((out_samp -1)*(dt_step-1)) ) + 1 ))]) ) > 0 ## check that the length of the initial sample range is positvie 

    #max_horizon = round(( t_resolution - input_time_stamps)//output_time_stamps)
    horihori = dt_step*(input_time_stamps+(horizons[0]*input_time_stamps)-1)+1
    random_samples = torch.randint(len(t_sample_space)-horihori, (t_iteration,))

    for s in range(t_iteration):
        #import pdb; pdb.set_trace()
        count_t_iter += 1
        
        for out_samp in range(len(n_tsamples)):
            #import pdb; pdb.set_trace()

            tsamples = n_tsamples[out_samp]
            horizon = horizons[out_samp] #(tsamples-output_time_stamps)//output_time_stamps

            # p.print(f"tsample: {tsamples} ")
            # p.print(f"horizon: {horizon}")
            

            if dynamic_loss_weight_per_fpass:
                f_pass_weights = dynamic_weight_loss_sq(count_t_iter, total_iter, dynamic_loss_weight_per_fpass_constant_parameter, max_horizon, horizon).to(device)
            elif dynamic_loss_weight_per_fpass == None:
                raise TypeError("Specify dynamic_w_l_f_pass (True or False) ")

            if dynamic_loss_weight_per_fpass_reversed:
                f_pass_weights = dynamic_weight_loss_sq(total_iter- count_t_iter + 1, total_iter,dynamic_loss_weight_per_fpass_constant_parameter, max_horizon, horizon).to(device)
                f_pass_weights = torch.flip(f_pass_weights, dims=[0])
            elif dynamic_loss_weight_per_fpass_reversed == None:
                raise TypeError("Specify dynamic_w_l_f_pass_reversed (True or False) ")

            if dynamic_loss_weight_per_tstamp:
                t_step_weights = dynamic_weight_loss(count_t_iter, total_iter, dynamic_loss_weight_per_tstamp_constant_parameter,  output_time_stamps  ).to(device)
            elif dynamic_loss_weight_per_tstamp == None:
                raise TypeError("Specify dynamic_w_l_t_steps (True or False) ")


            #import pdb; pdb.set_trace()

            for (data, u_super, x, parameters) in train_loader:
                #import pdb; pdb.set_trace()

                #import pdb; pdb.set_trace()
                data = data.to(device)
                parameters = parameters[...,:args.no_parameters].to(device)

                time_sampling_choice = args.time_sampling_choice
                
                # data_batch = batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
                # p.print(f"data_batch.indicies: {data_batch.indicies.shape}")
                # p.print(f"data_batch.indicies: {data_batch.indicies}")
                # time_indicies = t_sample_space[data_batch.indicies]


                indicies = torch.arange(random_samples[s], random_samples[s]+horihori,dt_step ).to(device)
                #p.print(f"indicies: {indicies.shape}")
                #p.print(f"indicies: {indicies}")
                time_indicies = t_sample_space[indicies.unsqueeze(0)]


                

                xy = torch.gather(data, -1, time_indicies.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
                xy_t = torch.ones_like(xy)[:,0,:].to(device)
                xy_t = xy_t*timestamps[time_indicies]
                xy_t = xy_t.unsqueeze(1).repeat(1,data.shape[1],1)
                xy_tindicies = time_indicies.long()

                #time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
                time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
        
                x = xy[..., :input_time_stamps ]
                x_t = xy_t[..., :input_time_stamps ]
                x_tindicies = xy_tindicies[..., :input_time_stamps ]

                # if s < 5:
                #     print("time_stamps->", time_stamps)
                #     print("starting_index->", x_tindicies)


                
                loss = 0
                a_l = 0


                #import pdb; pdb.set_trace()
                f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)

                if dynamic_loss_weight_per_fpass:
                    
                    # if args.dynamic_loss_weight_per_fpass_type.startswith("global"):
                    #     random_steps = time_indicies[:,output_time_stamps]
                    #     f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
                    #     random_steps_dx = [ torch.div(irx, output_time_stamps, rounding_mode='floor')-1 for irx in random_steps]
                    #     for irx in range(len(random_steps_dx)):
                    #         f_pass_weights_random[irx,:] = torch.tensor([f_pass_weights[irx].item() for irx in range(random_steps_dx[irx],random_steps_dx[irx]+horizon)]).to(device)

                    if dynamic_loss_weight_per_fpass_type.startswith("global"):
                        random_steps = time_indicies[:,output_time_stamps:][:,::output_time_stamps]
                        f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
                        for irx in range(args.batch_size_train):
                            random_steps_dx = [ torch.div(irx-output_time_stamps, output_time_stamps, rounding_mode='floor').item() for irx in random_steps[irx]]
                            f_pass_weights_random[irx,:] = torch.tensor( [f_pass_weights[irx].item() for irx in random_steps_dx] ).to(device)
  

                    elif dynamic_loss_weight_per_fpass_type.startswith("local"):
                        f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)
                


                if random_horizon:
                    horizon_range = torch.arange(1,horizon+1)
                    rand_horizon = horizon_range[torch.randint(horizon_range.size(0), size=(1,))].item()
                else:
                    rand_horizon = horizon


                if push_forward:
                    if push_forward_parameter_random:
                        horizon_grad_range = torch.arange(max(1, rand_horizon-push_forward_parameter),rand_horizon+1)
                        rand_horizon_grad = horizon_grad_range[torch.randint(horizon_grad_range.size(0), size=(1,))].item()
                    else:
                        rand_horizon_grad = max(1, rand_horizon-push_forward_parameter)
                else:
                    rand_horizon_grad = rand_horizon



                # import pdb; pdb.set_trace()

                with torch.no_grad():    
                    for t in range(rand_horizon-rand_horizon_grad):
                        #import pdb; pdb.set_trace()
                        #import pdb; pdb.set_trace()

                        # y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                        # y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                        # y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

                        y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                        y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                        y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]


                        if time_prediction == "constant":
                            if args.dataset_name == "E1":
                                out = model(x).to(device)
                            elif args.dataset_name == "E2":
                                out = model(torch.cat((x, parameters), dim=-1)).to(device)
                            elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
                                out = model(torch.cat((x, parameters), dim=-1)).to(device)


                        if time_prediction == "variable":

                            if time_conditioning == "addition":
                                x_x_t = x + x_t
                                if args.dataset_name == "E1":
                                    out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                                elif args.dataset_name == "E2":
                                    out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                            elif time_conditioning == "concatenate":
                                if args.dataset_name == "E1":
                                    out = model( x, x_t, y_t ).to(device)
                                elif args.dataset_name == "E2":
                                    out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                            
                            
                            elif time_conditioning == "attention":
                                if args.dataset_name == "E1":
                                    out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                                elif args.dataset_name == "E2":
                                    out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


                        train_print_time(args, ep, last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )

                        #import pdb; pdb.set_trace()


                        if output_time_stamps > input_time_stamps:
                            x = out[...,-input_time_stamps:]
                            x_t = y_t[...,-input_time_stamps:]
                            x_tindicies = y_tindicies[...,-input_time_stamps:]

                        elif output_time_stamps == input_time_stamps:
                            x = torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                            x_t = torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                            x_tindicies = torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                        
                        elif output_time_stamps < input_time_stamps:
                            x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
                            x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                            x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                        
                        a_l += 1

                        #print("loss->", loss)


                #import pdb; pdb.set_trace()
                for t in range(rand_horizon-rand_horizon_grad, rand_horizon):
                    #import pdb; pdb.set_trace()
                    #import pdb; pdb.set_trace()

                    y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                    y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                    y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                

                    # if noise:
                    #     if norm:
                    #         x = normalizer(x)
                    #         x = x + torch.randn(x.shape, device=x.device) * noise_std
                    #     x = x  + torch.randn(x.shape, device=x.device) * torch.max(x)*noise_std


                    if norm:
                        x = normalizer(x)

                    if noise:
                        x = x  + torch.randn(x.shape, device=x.device) * torch.std(x)*noise_std


                    if time_prediction == "constant":
                        if args.dataset_name == "E1":
                            out = model(x).to(device)
                        elif args.dataset_name == "E2":
                            out = model(torch.cat((x, parameters), dim=-1)).to(device)
                        elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
                            out = model(torch.cat((x, parameters), dim=-1)).to(device)


                    #import pdb; pdb.set_trace()
                    if time_prediction == "variable":

                        if time_conditioning == "addition":
                            x_x_t = x + x_t
                            if args.dataset_name == "E1":
                                out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                            elif args.dataset_name == "E2":
                                out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                        elif time_conditioning == "concatenate":
                            if args.dataset_name == "E1":
                                out = model( x, x_t, y_t ).to(device)
                            elif args.dataset_name == "E2":
                                out = model( torch.cat((x, x_t,y_t, parameters), dim=-1 ) ).to(device)

                        elif time_conditioning == "attention":
                            if args.dataset_name == "E1":
                                out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                            elif args.dataset_name == "E2":
                                out = model(  x, x_tindicies, y_tindicies, parameters).to(device)
                    
                                
                    if norm:
                        out = normalizer.inverse(out)


                    #args.predict_difference = True
                    if args.predict_difference:
                        if args.dataset_name == "KS1":
                            out = x + 0.3*out
                        else:
                            out = x + out

                    #import pdb; pdb.set_trace()

                    loss_t = criterion(out, y).sum(dim=[1]).to(device)
                    loss_t = torch.sqrt(loss_t).to(device)

                    loss_t_w = t_step_weights*loss_t

                    loss += (f_pass_weights_random[:,a_l]*loss_t_w.sum(dim=[1])).sum()

                    #print("\n")
                    train_print_time(args, ep,last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )
                    #import pdb; pdb.set_trace()



                    curiculum_learning = False
                    #print("ep, epochs, k k_transtion -->",ep, args.epochs[proto], t, k_transition(ep, args.t_resolution, args.epochs[proto]) ) 
                    if curiculum_learning and t >= k_transition(ep, t_resolution, epochs):
                        
                        # if s % 500 == 0:
                        #     print("ep, epochs, k k_transtion -->",ep, args.epochs[proto], t, k_transition(ep, args.t_resolution, args.epochs[proto]) )
                            #print("switching to true solution")
                        out = y


                        
                    if output_time_stamps > input_time_stamps:
                        x = out[...,-input_time_stamps:]
                        x_t = y_t[...,-input_time_stamps:]
                        x_tindicies = y_tindicies[...,-input_time_stamps:]

                    elif output_time_stamps == input_time_stamps:
                        x = torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                        x_t = torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                        x_tindicies = torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                    
                    elif output_time_stamps < input_time_stamps:
                        x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
                        x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                        x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                    
                    a_l += 1


                train_l2_full += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        if sheduler_change == "iteration":
            #p.print(f"learning_rate: { optimizer.param_groups[0]['lr']}" )
            scheduler.step()


        #import pdb; pdb.set_trace()
        if (count_t_iter) % 500 == 0:
            p.print(f"t_iter: {count_t_iter}/{total_iter}")
            p.print(f"f_pass_weights: {f_pass_weights}")
            p.print(f"f_pass_weights_random: {f_pass_weights_random[:3]}")
            p.print("\n")
            
        
    return train_l2_full, model, count_t_iter






def random_time_sampling_splits(
    args,
    count_t_iter,
    proto,

    ep,
    last_epoch_no,

    t_iteration,
    n_tsamples,
    data_batchsize,

    model,
    optimizer,
    train_loader,
    criterion,


    input_time_stamps,
    output_time_stamps,
    t_resolution, 

    timestamps,

    f_pass_weights,
    t_step_weights,

    time_prediction,
    time_conditioning,

    horizon,

    dt_step,

    noise,
    noise_std,
    norm,

    ):


    train_l2_full = 0
    

    #import pdb; pdb.set_trace()

    total_iter = sum(args.epochs)*t_iteration
    t_sample_space = torch.arange(t_resolution)

    if time_prediction == "constant":
        assert args.time_sampling_choice == 1  # use the right type of random time generator
        assert dt_step-1 <= int( t_resolution/max(n_tsamples) )

    # elif time_prediction == "variable":
    #     assert args.time_sampling_choice > 1

    for out_samp in range(len(n_tsamples)):
        assert len(torch.tensor([t for t in range(0, (args.t_resolution -  (out_samp + ((out_samp -1)*(dt_step-1)) ) + 1 ))]) ) > 0 ## check that the length of the initial sample range is positvie 


    for s in range(t_iteration):
        #import pdb; pdb.set_trace()
        count_t_iter += 1

        for out_samp in range(len(n_tsamples)):

            tsamples = n_tsamples[out_samp]
            horizon = args.horizon[out_samp] #(tsamples-output_time_stamps)//output_time_stamps

            # p.print(f"tsample: {tsamples} ")
            # p.print(f"horizon: {horizon}")


            if args.dynamic_loss_weight_per_fpass[proto]:
                f_pass_weights = dynamic_weight_loss_sq(count_t_iter, total_iter, args.dynamic_loss_weight_per_fpass_constant_parameter[proto], (args.t_resolution - args.output_time_stamps)//args.output_time_stamps, args.horizon).to(device)
            elif args.dynamic_loss_weight_per_fpass[proto] == None:
                raise TypeError("Specify dynamic_w_l_f_pass (True or False) ")

            if args.dynamic_loss_weight_per_fpass_reversed[proto]:
                f_pass_weights = dynamic_weight_loss_sq(total_iter- count_t_iter + 1, total_iter,args.dynamic_loss_weight_per_fpass_constant_parameter[proto], (args.t_resolution - args.output_time_stamps)//args.output_time_stamps, args.horizon).to(device)
                f_pass_weights = torch.flip(f_pass_weights, dims=[0])
            elif args.dynamic_loss_weight_per_fpass_reversed[proto] == None:
                raise TypeError("Specify dynamic_w_l_f_pass_reversed (True or False) ")

            if args.dynamic_loss_weight_per_tstamp[proto]:
                t_step_weights = dynamic_weight_loss(count_t_iter, total_iter, args.dynamic_loss_weight_per_tstamp_constant_parameter[proto],  args.output_time_stamps  ).to(device)
            elif args.dynamic_loss_weight_per_tstamp[proto] == None:
                raise TypeError("Specify dynamic_w_l_t_steps (True or False) ")


            #import pdb; pdb.set_trace()sss
            noise_sampler = torch.randint(2,(70,1))
            batch_counter = 0

            for (data, u_super, x, parameters) in train_loader:

                batch_counter += 1
                #import pdb; pdb.set_trace()
                data = data.to(device)
                parameters = parameters[...,:args.no_parameters].to(device)

                time_sampling_choice = args.time_sampling_choice
                data_batch = batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
                time_indicies = t_sample_space[data_batch.indicies]

                xy = torch.gather(data, -1, time_indicies.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
                xy_t = torch.ones_like(xy)[:,0,:].to(device)
                xy_t = xy_t*timestamps[time_indicies]
                xy_t = xy_t.unsqueeze(1).repeat(1,data.shape[1],1)
                xy_tindicies = time_indicies.long()

                #time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
                time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
        
                x = xy[..., :input_time_stamps ]
                x_t = xy_t[..., :input_time_stamps ]
                x_tindicies = xy_tindicies[..., :input_time_stamps ]

                # if s < 5:
                #     print("time_stamps->", time_stamps)
                #     print("starting_index->", x_tindicies)



                #import pdb; pdb.set_trace()
                time_indicies_zero = torch.zeros( (data.shape[0], 1) ).to(torch.long)
                x0 = torch.gather(data, -1, time_indicies_zero.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
                x0_t = torch.ones_like(x0)[:,0,:].to(device)
                x0_t = x0_t*timestamps[time_indicies_zero]
                x0_t = x0_t.unsqueeze(1).repeat(1,data.shape[1],1)
                xy_tindicies_zero = time_indicies_zero.long()



                loss = 0
                a_l = 0


                #import pdb; pdb.set_trace()
                f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)

                if args.dynamic_loss_weight_per_fpass[proto]:
                    
                    # if args.dynamic_loss_weight_per_fpass_type.startswith("global"):
                    #     random_steps = time_indicies[:,output_time_stamps]
                    #     f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
                    #     random_steps_dx = [ torch.div(irx, output_time_stamps, rounding_mode='floor')-1 for irx in random_steps]
                    #     for irx in range(len(random_steps_dx)):
                    #         f_pass_weights_random[irx,:] = torch.tensor([f_pass_weights[irx].item() for irx in range(random_steps_dx[irx],random_steps_dx[irx]+horizon)]).to(device)

                    if args.dynamic_loss_weight_per_fpass_type.startswith("global"):
                        random_steps = time_indicies[:,output_time_stamps:][:,::output_time_stamps]
                        f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
                        for irx in range(args.batch_size_train):
                            random_steps_dx = [ torch.div(irx-output_time_stamps, output_time_stamps, rounding_mode='floor').item() for irx in random_steps[irx]]
                            f_pass_weights_random[irx,:] = torch.tensor( [f_pass_weights[irx].item() for irx in random_steps_dx] ).to(device)
  

                    elif args.dynamic_loss_weight_per_fpass_type.startswith("local"):
                        f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)
                


                if args.random_horizon[proto]:
                    horizon_range = torch.arange(1,horizon+1)
                    rand_horizon = horizon_range[torch.randint(horizon_range.size(0), size=(1,))].item()
                else:
                    rand_horizon = horizon


                if args.push_forward[proto]:
                    if args.push_forward_parameter_random[proto]:
                        horizon_grad_range = torch.arange(max(1, rand_horizon-args.push_forward_parameter[proto]),rand_horizon+1)
                        rand_horizon_grad = horizon_grad_range[torch.randint(horizon_grad_range.size(0), size=(1,))].item()
                    else:
                        rand_horizon_grad = max(1, rand_horizon-args.push_forward_parameter[proto])
                else:
                    rand_horizon_grad = rand_horizon



                # import pdb; pdb.set_trace()

                with torch.no_grad():    
                    for t in range(rand_horizon-rand_horizon_grad):
                        #import pdb; pdb.set_trace()

                        # y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                        # y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                        # y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

                        y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                        y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                        y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]


                        if time_prediction.startswith("constant"):
                            if args.dataset_name.endswith("E1"):
                                out = model(x).to(device)
                            elif args.dataset_name.endswith("E2"):
                                out = model(torch.cat((x, parameters), dim=-1)).to(device)


                        if time_prediction.startswith("variable"):

                            if time_conditioning.startswith("addition"):
                                x_x_t = x + x_t
                                if args.dataset_name.endswith("E1"):
                                    out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                                elif args.dataset_name.endswith("E2"):
                                    out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                            elif time_conditioning.startswith("concatenate"):
                                if args.dataset_name.endswith("E1"):
                                    out = model( x, x_t, y_t ).to(device)
                                elif args.dataset_name.endswith("E2"):
                                    out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                            
                            
                            elif time_conditioning.startswith("attention"):
                                if args.dataset_name.endswith("E1"):
                                    out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                                elif args.dataset_name.endswith("E2"):
                                    out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


                        train_print_time(args, ep, last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )

                        #import pdb; pdb.set_trace()


                        if output_time_stamps > input_time_stamps:
                            x = out[...,-input_time_stamps:]
                            x_t = y_t[...,-input_time_stamps:]
                            x_tindicies = y_tindicies[...,-input_time_stamps:]

                        elif output_time_stamps == input_time_stamps:
                            x = torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                            x_t = torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                            x_tindicies = torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                        
                        elif output_time_stamps < input_time_stamps:
                            x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
                            x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                            x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                        
                        a_l += 1

                        #print("loss->", loss)


                #import pdb; pdb.set_trace()
                for t in range(rand_horizon-rand_horizon_grad, rand_horizon):
                    #import pdb; pdb.set_trace()

                    y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                    y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                    y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                

                    # if noise:
                    #     if norm:
                    #         x = normalizer(x)
                    #         x = x + torch.randn(x.shape, device=x.device) * noise_std
                    #     x = x  + torch.randn(x.shape, device=x.device) * torch.max(x)*noise_std


                    if norm:
                        x = normalizer(x)

                    if noise:
                        if noise_sampler[batch_counter] == 1:
                            x = x  + torch.randn(x.shape, device=x.device) * torch.std(x)*noise_std

                    #import pdb; pdb.set_trace()
                    if time_prediction.startswith("constant"):
                        if args.dataset_name.endswith("E1"):
                            #x = torch.cat((x0, x), dim=-1)
                            out = model(torch.cat((x0, x), dim=-1)).to(device)
                        elif args.dataset_name.endswith("E2"):
                            out = model(torch.cat((x, parameters), dim=-1)).to(device)

                    #import pdb; pdb.set_trace()
                    if time_prediction.startswith("variable"):

                        if time_conditioning.startswith("addition"):
                            x_x_t = x + x_t
                            if args.dataset_name.endswith("E1"):
                                out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                            elif args.dataset_name.endswith("E2"):
                                out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                        elif time_conditioning.startswith("concatenate"):
                            if args.dataset_name.endswith("E1"):
                                out = model( x, x_t, y_t ).to(device)
                            elif args.dataset_name.endswith("E2"):
                                out = model( torch.cat((x, x_t,y_t, parameters), dim=-1 ) ).to(device)

                        elif time_conditioning.startswith("attention"):
                            if args.dataset_name.endswith("E1"):
                                out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                            elif args.dataset_name.endswith("E2"):
                                out = model(  x, x_tindicies, y_tindicies, parameters).to(device)
                    
                                
                    if norm:
                        out = normalizer.inverse(out)


                    #import pdb; pdb.set_trace()

                    # loss_t = criterion(out, y).sum(dim=[1]).to(device)
                    # loss_t = criterion(out, y).sum(dim=[1]).to(device)
                    # loss_t = torch.sqrt(loss_t).to(device)

                    # loss_t_w = t_step_weights*loss_t

                    # loss += (f_pass_weights_random[:,a_l]*loss_t_w.sum(dim=[1])).sum()
                    #import pdb; pdb.set_trace()

                    loss += criterion(out,x_t[:,0,:]).sum()

                    # if s == 0:
                    #     print("prediction ->", out.squeeze())
                    #     print("true -->", x_t[:,0,0])
                    #     print("loss -->", loss)
                    #     print("\n")

                    train_print_time(args, ep,last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )
                    #import pdb; pdb.set_trace()


                    cur_learn = False

                    #print("ep, epochs, k k_transtion -->",ep, args.epochs[proto], t, k_transition(ep, args.t_resolution, args.epochs[proto]) )
                          
                    if cur_learn and t >= k_transition(ep, args.t_resolution, args.epochs[proto]):
                        # if s % 500 == 0:
                        #     print("ep, epochs, k k_transtion -->",ep, args.epochs[proto], t, k_transition(ep, args.t_resolution, args.epochs[proto]) )
                            #print("switching to true solution")
                        out = y
                        

                    # if output_time_stamps > input_time_stamps:
                    #     x = out[...,-input_time_stamps:]
                    #     x_t = y_t[...,-input_time_stamps:]
                    #     x_tindicies = y_tindicies[...,-input_time_stamps:]

                    # elif output_time_stamps == input_time_stamps:
                    #     x =  out
                    #     x_t =  y_t
                    #     x_tindicies = y_tindicies
                    
                    # elif output_time_stamps < input_time_stamps:
                    #     x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
                    #     x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                    #     x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                    
                    a_l += 1


                train_l2_full += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #import pdb; pdb.set_trace()
        if (count_t_iter) % 500 == 0:
            p.print(f"t_iter: {count_t_iter}/{total_iter}")
            p.print(f"f_pass_weights: {f_pass_weights}")
            p.print(f"f_pass_weights_random: {f_pass_weights_random[:3]}")
            p.print("\n")
            
        
    return train_l2_full, model, count_t_iter





































def sequential_time_sampling(
    args,
    count_t_iter,
    proto,

    ep,
    epochs,
    last_epoch_no,

    t_iteration,
    n_tsamples,
    data_batchsize,

    model,
    optimizer,
    train_loader,
    criterion,


    input_time_stamps,
    output_time_stamps,
    t_resolution, 

    timestamps,

    f_pass_weights,
    t_step_weights,

    time_prediction,
    time_conditioning,

    max_horizon,
    horizons,
    random_horizon,
    

    dt_step,

    noise,
    noise_std,
    norm,

    scheduler,
    sheduler_change,


    dynamic_loss_weight_per_fpass,
    dynamic_loss_weight_per_fpass_type,
    dynamic_loss_weight_per_fpass_reversed,
    dynamic_loss_weight_per_fpass_constant_parameter,

    dynamic_loss_weight_per_tstamp,
    dynamic_loss_weight_per_tstamp_constant_parameter,

    push_forward,
    push_forward_parameter_random,
    push_forward_parameter,

    ):


    train_l2_full = 0
    

    #import pdb; pdb.set_trace()

    total_iter = epochs*t_iteration
    t_sample_space = torch.arange(t_resolution)

    if time_prediction == "constant":
        assert args.time_sampling_choice == 1 or args.time_sampling_choice == 5 # use the right type of random time generator
        assert dt_step-1 <= int( t_resolution/max(n_tsamples) )

    # elif time_prediction == "variable":
    #     assert args.time_sampling_choice > 1

    for out_samp in range(len(n_tsamples)):
        assert len(torch.tensor([t for t in range(0, (t_resolution -  (out_samp + ((out_samp -1)*(dt_step-1)) ) + 1 ))]) ) > 0 ## check that the length of the initial sample range is positvie 

    #max_horizon = round(( t_resolution - input_time_stamps)//output_time_stamps)

    for s in range(t_iteration):
        #import pdb; pdb.set_trace()
        count_t_iter += 1

        for out_samp in range(len(n_tsamples)):
            #import pdb; pdb.set_trace()

            tsamples = n_tsamples[out_samp]
            horizon = horizons[out_samp] #(tsamples-output_time_stamps)//output_time_stamps

            # p.print(f"tsample: {tsamples} ")
            # p.print(f"horizon: {horizon}")
            

            if dynamic_loss_weight_per_fpass:
                f_pass_weights = dynamic_weight_loss_sq(count_t_iter, total_iter, dynamic_loss_weight_per_fpass_constant_parameter, max_horizon, horizon).to(device)
            elif dynamic_loss_weight_per_fpass == None:
                raise TypeError("Specify dynamic_w_l_f_pass (True or False) ")

            if dynamic_loss_weight_per_fpass_reversed:
                f_pass_weights = dynamic_weight_loss_sq(total_iter- count_t_iter + 1, total_iter,dynamic_loss_weight_per_fpass_constant_parameter, max_horizon, horizon).to(device)
                f_pass_weights = torch.flip(f_pass_weights, dims=[0])
            elif dynamic_loss_weight_per_fpass_reversed == None:
                raise TypeError("Specify dynamic_w_l_f_pass_reversed (True or False) ")

            if dynamic_loss_weight_per_tstamp:
                t_step_weights = dynamic_weight_loss(count_t_iter, total_iter, dynamic_loss_weight_per_tstamp_constant_parameter,  output_time_stamps  ).to(device)
            elif dynamic_loss_weight_per_tstamp == None:
                raise TypeError("Specify dynamic_w_l_t_steps (True or False) ")


            #import pdb; pdb.set_trace()

            for bb, (data, u_super, x, parameters) in enumerate(train_loader):
                #import pdb; pdb.set_trace()

                #import pdb; pdb.set_trace()
                data = data.to(device)
                parameters = parameters[...,:args.no_parameters].to(device)

                time_sampling_choice = args.time_sampling_choice
                data_batch = batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
                time_indicies = t_sample_space[data_batch.indicies]
                xy = torch.gather(data, -1, time_indicies.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
                xy_t = torch.ones_like(xy)[:,0,:].to(device)
                xy_t = xy_t*timestamps[time_indicies]

                xy_t = torch.cat((torch.diff(xy_t, dim=-1), torch.zeros(xy_t.shape[0], 1).to(device)), dim=-1)

                xy_t = xy_t.unsqueeze(1).repeat(1, data.shape[1],1)
                xy_tindicies = time_indicies.long()

                #time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
                time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
        
                x = xy[..., :input_time_stamps ]
                x_t = xy_t[..., :input_time_stamps ]
                x_tindicies = xy_tindicies[..., :input_time_stamps ]

                # if s < 5:
                #     print("time_stamps->", time_stamps)
                #     print("starting_index->", x_tindicies)


                
                loss = 0
                a_l = 0


                #import pdb; pdb.set_trace()
                f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)

                if dynamic_loss_weight_per_fpass:
                    
                    # if args.dynamic_loss_weight_per_fpass_type.startswith("global"):
                    #     random_steps = time_indicies[:,output_time_stamps]
                    #     f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
                    #     random_steps_dx = [ torch.div(irx, output_time_stamps, rounding_mode='floor')-1 for irx in random_steps]
                    #     for irx in range(len(random_steps_dx)):
                    #         f_pass_weights_random[irx,:] = torch.tensor([f_pass_weights[irx].item() for irx in range(random_steps_dx[irx],random_steps_dx[irx]+horizon)]).to(device)

                    if dynamic_loss_weight_per_fpass_type.startswith("global"):
                        random_steps = time_indicies[:,output_time_stamps:][:,::output_time_stamps]
                        f_pass_weights_random = torch.ones(args.batch_size_train,horizon).to(device)
                        for irx in range(args.batch_size_train):
                            random_steps_dx = [ torch.div(irx-output_time_stamps, output_time_stamps, rounding_mode='floor').item() for irx in random_steps[irx]]
                            f_pass_weights_random[irx,:] = torch.tensor( [f_pass_weights[irx].item() for irx in random_steps_dx] ).to(device)
  

                    elif dynamic_loss_weight_per_fpass_type.startswith("local"):
                        f_pass_weights_random = f_pass_weights.unsqueeze(0).repeat(args.batch_size_train,1)
                


                if random_horizon:
                    horizon_range = torch.arange(1,horizon+1)
                    rand_horizon = horizon_range[torch.randint(horizon_range.size(0), size=(1,))].item()
                else:
                    rand_horizon = horizon


                if push_forward:
                    if push_forward_parameter_random:
                        horizon_grad_range = torch.arange(max(1, rand_horizon-push_forward_parameter),rand_horizon+1)
                        rand_horizon_grad = horizon_grad_range[torch.randint(horizon_grad_range.size(0), size=(1,))].item()
                    else:
                        rand_horizon_grad = max(1, rand_horizon-push_forward_parameter)
                else:
                    rand_horizon_grad = rand_horizon



                # import pdb; pdb.set_trace()

                # with torch.no_grad():    
                #     for t in range(rand_horizon-rand_horizon_grad):
                #         #import pdb; pdb.set_trace()
                #         #import pdb; pdb.set_trace()

                #         # y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                #         # y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                #         # y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

                #         y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                #         y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                #         y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]


                #         if time_prediction == "constant":
                #             if args.dataset_name == "E1":
                #                 out = model(x).to(device)
                #             elif args.dataset_name == "E2":
                #                 out = model(torch.cat((x, parameters), dim=-1)).to(device)
                #             elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
                #                 out = model(torch.cat((x, parameters), dim=-1)).to(device)


                #         if time_prediction == "variable":

                #             if time_conditioning == "addition":
                #                 x_x_t = x + x_t
                #                 if args.dataset_name == "E1":
                #                     out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                #                 elif args.dataset_name == "E2":
                #                     out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                #             elif time_conditioning == "concatenate":
                #                 if args.dataset_name == "E1":
                #                     out = model( x, x_t, y_t ).to(device)
                #                 elif args.dataset_name == "E2":
                #                     out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                            
                            
                #             elif time_conditioning == "attention":
                #                 if args.dataset_name == "E1":
                #                     out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                #                 elif args.dataset_name == "E2":
                #                     out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


                #         train_print_time(args, ep, last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )

                #         #import pdb; pdb.set_trace()


                #         if output_time_stamps > input_time_stamps:
                #             x = out[...,-input_time_stamps:]
                #             x_t = y_t[...,-input_time_stamps:]
                #             x_tindicies = y_tindicies[...,-input_time_stamps:]

                #         elif output_time_stamps == input_time_stamps:
                #             x = torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                #             x_t = torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                #             x_tindicies = torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                        
                #         elif output_time_stamps < input_time_stamps:
                #             x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
                #             x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                #             x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                        
                #         a_l += 1

                        #print("loss->", loss)


                #import pdb; pdb.set_trace()
                teacher_forcing_count = 0
                for t in range(rand_horizon-rand_horizon_grad, rand_horizon):
                    #import pdb; pdb.set_trace()
                    #import pdb; pdb.set_trace()
                    #if s == 0 and bb == 0:
                        #p.print(f"t --> {t}")

                    y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                    y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                    y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                

                    # if noise:
                    #     if norm:
                    #         x = normalizer(x)
                    #         x = x + torch.randn(x.shape, device=x.device) * noise_std
                    #     x = x  + torch.randn(x.shape, device=x.device) * torch.max(x)*noise_std


                    if norm:
                        #p.print("Normalising input.....")
                        x = normalizer(x)

                    if noise and s == 0 and bb == 0 and teacher_forcing_count > 0:
                        #p.print("Adding noise........")
                        x = x  + torch.randn(x.shape, device=x.device) * torch.std(x)*noise_std


                    if time_prediction == "constant":
                        if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                            out = model(x).to(device)
                        elif args.dataset_name == "E2":
                            out = model(torch.cat((x, parameters), dim=-1)).to(device)
                        elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
                            out = model(torch.cat((x, parameters), dim=-1)).to(device)


                    #import pdb; pdb.set_trace()
                    if time_prediction == "variable":

                        if time_conditioning == "addition":
                            x_x_t = x + x_t
                            if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                                out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                            elif args.dataset_name == "E2":
                                out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                        elif time_conditioning == "concatenate":
                            if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                                out = model( x, x_t, y_t ).to(device)
                            elif args.dataset_name == "E2":
                                out = model( torch.cat((x, x_t, y_t, parameters), dim=-1 ) ).to(device)

                        elif time_conditioning == "attention":
                            if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                                #x.permute(0,2,1).to(device), y.permute(0,2,1).to(device),x_mask, y_mask, x_t[:, 0, :].to(device), y_t[:, 0, :].to(device)
                                #print("x, x_t, y_t -->", x.shape, x_t.shape, y_t.shape)
                                out = model( x, x_t, y_t ).to(device)
                                #print("y, out", y.shape, out.shape)

                                #out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                            elif args.dataset_name == "E2":
                                out = model(  x, x_tindicies, y_tindicies, parameters).to(device)
                    
                                
                    if norm:
                        out = normalizer.inverse(out)

 
                    #args.predict_difference = True
                    if args.predict_difference:
                        if args.dataset_name == "KS1":
                            out = x + 0.3*out
                        else:
                            out = x + out

                    #import pdb; pdb.set_trace()



                    # loss += loss_t


                    # #p.print(f"loss_t --> { criterion(out, y).shape}")
                    # loss_t = criterion(out, y).sum(dim=[1]).to(device)

                    # #p.print(f"loss_t --> {loss_t.shape}")
                    # #loss_t = torch.sqrt(loss_t).to(device)

                    # loss_t_w = t_step_weights*loss_t

                    # loss += (f_pass_weights_random[:,a_l]*loss_t_w.sum(dim=[1])).sum()                    





                    loss_t = criterion(out, y).to(device)
                    loss_t = torch.sqrt(loss_t).to(device)

                    #p.print(f"loss_t, t_step_weights:  {loss_t.shape}, {t_step_weights.shape}")
                    loss_t_w = t_step_weights*loss_t

                    #p.print(f"loss_t_w, t_step_weights:  {loss_t_w.shape}, {f_pass_weights_random[:,a_l].shape}")
                    loss += (f_pass_weights_random[:,a_l]*loss_t_w.sum(dim=[1])).sum()



                    #print("\n")
                    train_print_time(args, ep,last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )
                    #import pdb; pdb.set_trace()

                    curiculum_learning = False # True #True
                    #p.print("Doing curriculum........")
                    #p.print(f"ep, epochs, t, t_resolution, epochs --> {ep}, {args.epochs[proto]}, {t}, {len(range(rand_horizon-rand_horizon_grad, rand_horizon))}, {args.epochs[proto]}")  
                    #p.print(f"k_trans: {k_transition(ep, len(range(rand_horizon-rand_horizon_grad, rand_horizon)), args.epochs[proto])}" )
                    
                    if curiculum_learning and t >= k_transition(ep, len(range(rand_horizon-rand_horizon_grad, rand_horizon)), epochs):
                        
                        if s == 0 and bb == 0 and teacher_forcing_count == 0:
                            p.print("Transition zone (changing to teacher-forcing....)")
                            p.print(f"t: {t}")
                            p.print(f"t_trans: {k_transition(ep, len(range(rand_horizon-rand_horizon_grad, rand_horizon)), args.epochs[proto])}" ) 
                        
                        out = y

                        teacher_forcing_count = teacher_forcing_count + 1
                        
                        # if s % 100 == 0:
                        # #     print("ep, epochs, k k_transtion -->",ep, args.epochs[proto], t, k_transition(ep, args.t_resolution, args.epochs[proto]) )
                        #     p.print(f"k_trans: {k_transition(ep, len(range(rand_horizon-rand_horizon_grad, rand_horizon)), args.epochs[proto])}" ) 
                        #     print("switching to teacher-forcing")

                        
                    

                        
                    if output_time_stamps > input_time_stamps:
                        x = out[...,-input_time_stamps:]
                        x_t = y_t[...,-input_time_stamps:]
                        x_tindicies = y_tindicies[...,-input_time_stamps:]

                    elif output_time_stamps == input_time_stamps:
                        x = torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                        x_t = torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                        x_tindicies = torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                    
                    elif output_time_stamps < input_time_stamps:
                        x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
                        x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                        x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                    
                    a_l += 1


                train_l2_full += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        if sheduler_change == "iteration":
            #p.print(f"learning_rate: { optimizer.param_groups[0]['lr']}" )
            scheduler.step()


        #import pdb; pdb.set_trace()
        if (count_t_iter) % 500 == 0:
            p.print(f"t_iter: {count_t_iter}/{total_iter}")
            p.print(f"f_pass_weights: {f_pass_weights}")
            p.print(f"f_pass_weights_random: {f_pass_weights_random[:3]}")
            p.print("\n")
            
        
    return train_l2_full/(t_iteration*(bb+1)), model, count_t_iter

