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
import time

from variable_autoregression.util import LpLoss, Printer, get_time, count_params, set_seed, return_checkpoint, dynamic_weight_loss, dynamic_weight_loss_sq, create_current_results_folder, load_auguments, save_config_file, create_data, create_next_data, batch_time_sampling, train_print_time, Normalizer_1D, k_transition, subsequent_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p = Printer(n_digits=6)




normalizer = Normalizer_1D()







# def run_epoch(
#     data_iter,
#     model,
#     loss_compute,
#     optimizer,
#     scheduler,
#     train_state,
# ):
#     """Train a single epoch"""
#     # print(scheduler[0])
#     # print(scheduler[1])
#     #print(time)
#     start = time.time()
#     total_tokens = 0
#     total_loss = 0
#     train_state.step += 1
#     for i, batch in enumerate(data_iter):
#         optimizer.zero_grad()
#         out = model.forward(
#             batch.input, batch.output, batch.input_mask, batch.output_mask, batch.input_time, batch.output_time,
#             batch.modes_in, batch.modes_out
#         )
#         loss = loss_compute(out, batch.output)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss
#         lr = optimizer.param_groups[0]["lr"]
#         elapsed = time.time() - start

#     print(
#         (
#             "Epoch Step: %6d | Loss: %6.5f "
#             +  "| Learning Rate: %6.1e"
#         )
#         % (train_state.step, total_loss/((i+1)*out.shape[0]), lr)
#     )

#     start = time.time()
#     scheduler.step()

#     return total_loss/((i+1)*out.shape[0]), train_state, out










def run_epoch(
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

    for s in range(t_iteration):
        #import pdb; pdb.set_trace()
        count_t_iter += 1

        for out_samp in range(len(n_tsamples)):
            #import pdb; pdb.set_trace()

            tsamples = n_tsamples[out_samp]
            horizon = horizons[out_samp] #(tsamples-output_time_stamps)//output_time_stamps

            # p.print(f"tsample: {tsamples} ")
            # p.print(f"horizon: {horizon}")
            

            for (data, u_super, x, parameters) in train_loader:
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


                rand_horizon = horizon
                rand_horizon_grad = horizon

                for t in range(horizon):
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

                        if time_conditioning == "attention":
                            if args.dataset_name == "E1":
                                #out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)

                                x_mask = torch.ones((1, 1, 1, input_time_stamps) ).to(device)
                                y_mask = subsequent_mask(input_time_stamps).to(device)
                                out = model( x.permute(0,2,1).to(device), y.permute(0,2,1).to(device),x_mask, y_mask, x_t[:, 0, :].to(device), y_t[:, 0, :].to(device), 100, 100 ).to(device)
                                out = out.permute(0,2,1)
                            elif args.dataset_name == "E2":
                                pass
                                #out = model(  x, x_tindicies, y_tindicies, parameters).to(device)
                    
                                
                    if norm:
                        out = normalizer.inverse(out)


                    #args.predict_difference = True
                    if args.predict_difference:
                        if args.dataset_name == "KS1":
                            out = x + 0.3*out
                        else:
                            out = x + out

                    #import pdb; pdb.set_trace()

                    loss_t = criterion(out, y).to(device)
                    #loss_t = torch.sqrt(loss_t).to(device)

                    loss_t_w = t_step_weights*loss_t
                    f_pass_weights_random = torch.ones((data_batchsize,1))
                    # loss += (f_pass_weights_random[:,0]*loss_t_w.sum(dim=[1])).sum()

                    loss += loss_t_w.sum()

                    #print("\n")
                    #train_print_time(args, ep,last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps  )
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
        if (count_t_iter) % 50 == 0:
            p.print(f"t_iter: {count_t_iter}/{total_iter}")
            p.print(f"f_pass_weights: {f_pass_weights}")
            p.print(f"f_pass_weights_random: {f_pass_weights_random[:3]}")
            p.print("\n")
            
        
    return train_l2_full, model, count_t_iter


