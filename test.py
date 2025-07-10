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

# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from constant_autoregression.argparser import arg_parse 
#from variable_autoregression.dataset.load_dataset import Input_Batch, Output_Batch, no_of_output_space, data_random_batch
from constant_autoregression.util import LpLoss, Printer, load_auguments, test_batch_time_sampling, test_print_time, Normalizer_1D, subsequent_mask, normalized_rmse


normalizer = Normalizer_1D()
p = Printer(n_digits=6)
# args = arg_parse()
# args = load_auguments(args, "arguments")



def constant_rollout_test( args, model, loader, timestamps, dt_step, t_resolution, norm):

    model.eval()
    #input_time_stamps = args.input_time_stamps
    #t_resolution = args.t_resolution
    data_batchsize = args.batch_size_test
    input_time_stamps = args.input_time_stamps
    output_time_stamps = args.output_time_stamps
    time_sampling_choice = int(1)
    t_sample_space = torch.arange(t_resolution).to(device)

    #timestamps = torch.tensor(timestamps).to(device)
    #p.print(f"t_sample_space: {len(t_sample_space)}")
    tsamples = len(t_sample_space[::int(dt_step)])
    #p.print(f"tsamples: {tsamples}")

    #horizon = int((tsamples-input_time_stamps)/output_time_stamps)
    if (tsamples-input_time_stamps)%output_time_stamps  == 0:
        horizon = int((tsamples-input_time_stamps)/output_time_stamps)
    else:
        horizon = int((tsamples-input_time_stamps)/output_time_stamps) + 1

    #import pdb; pdb.set_trace()
    # p.print(f"dt_step: {dt_step}")
    # p.print(f"horizon_test: {horizon}")


    with torch.no_grad():
        

        for b, (data, u_super, x, parameters) in enumerate(loader):

            data = data.to(device)
            current_batch_size = data.size(0)
            parameters = parameters[...,:args.no_parameters].to(device)
            
            data_batch = test_batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(current_batch_size, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
            time_indicies = t_sample_space[data_batch.indicies]
            #p.print(f"data_batch: {data_batch.shape}")
            #p.print(f"time_indicies: {time_indicies.shape}")
            #print("\n")
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
            time_stamps = [i for i in range(0, time_indicies.shape[-1]+output_time_stamps, output_time_stamps)]
            
            # print("horizon", horizon)
            # print("time_indicies", time_indicies.shape)
            # print("time_stamps", time_stamps)
            # x = xy[..., :output_time_stamps ]
            # x_t = xy_t[..., :output_time_stamps ]
            # x_tindicies = xy_tindicies[..., :output_time_stamps ]

            # x = xy[..., :input_time_stamps ]
            # x_t = xy_t[..., :input_time_stamps ]
            # x_tindicies = xy_tindicies[..., :input_time_stamps ]
            #import pdb; pdb.set_trace()


            if b == 0:
                train_actual = xy[..., input_time_stamps:].clone()
            else:
                train_actual = torch.cat((train_actual, xy[..., input_time_stamps:]), dim=0)

            # input:
            x = xy[..., : input_time_stamps]
            x_t = xy_t[..., :input_time_stamps]
            x_tindicies = xy_tindicies[..., :input_time_stamps ]
            
            #print("xy -->", xy.shape)
            #import pdb; pdb.set_trace()
            for t in range(horizon):
                #p.print(f"horizon: {t}")
                #import pdb; pdb.set_trace()
                #import pdb; pdb.set_trace()


                # y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                # y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                # y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]
                #print("in_range ->", )
                # print("t ->",t)
                # print("out_range ->", input_time_stamps+time_stamps[t],input_time_stamps+time_stamps[t+1])
                # print("\n")
                y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]

                #print("args.norm -->", args.norm)

                if norm:
                    p.print(f"doing normalisation: {norm}")
                    x = normalizer(x)  


                if args.time_prediction == "constant":
                    if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                        out = model(x).to(device)
                    elif args.dataset_name == "E2":
                        out = model(torch.cat((x, parameters), dim=-1)).to(device)
                    elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
                        out = model(torch.cat((x, parameters), dim=-1)).to(device)


                if args.time_prediction == "variable":

                    if args.time_conditioning == "addition":
                        x_x_t = x + x_t
                        if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                            out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                        elif args.dataset_name == "E2":
                            out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                    elif args.time_conditioning == "concatenate":
                        if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":

                            # if y_t.shape[-1] != output_time_stamps:
                            #     y_t = y_t.repeat(1,1,output_time_stamps)

                            #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")
                            if y_t.shape[-1] != output_time_stamps:
                                #y_t = y_t.repeat(1,1,output_time_stamps)
                                remain_step = int(output_time_stamps/y_t.shape[-1]) + 1
                                y_t = y_t.repeat(1,1,remain_step*y_t.shape[-1])[...,:output_time_stamps]
                                #y_t = torch.cat((y_t, y_t[..., :remain_step]), dim=-1)
                                #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")

                            #p.print(f"x, x_t, y_t:  {x.shape, x_t.shape, y_t.shape}")
                            out = model( x, x_t, y_t ).to(device)
                        elif args.dataset_name == "E2":
                            out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                    
                    
                    elif args.time_conditioning == "attention":
                        if args.dataset_name == "E1" or "B1" or "A1":
                            #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")

                            # if y_t.shape[-1] != output_time_stamps:
                            #     y_t = y_t.repeat(1,1,output_time_stamps)
                            #     #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")


                            #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")
                            if y_t.shape[-1] != output_time_stamps:
                                #y_t = y_t.repeat(1,1,output_time_stamps)
                                remain_step = int(output_time_stamps/y_t.shape[-1]) + 1
                                y_t = y_t.repeat(1,1,remain_step*y_t.shape[-1])[...,:output_time_stamps]
                                #y_t = torch.cat((y_t, y_t[..., :remain_step]), dim=-1)
                                #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")

                            out = model( x, x_t, y_t ).to(device)
                            #import pdb; pdb.set_trace()
                            #out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                            # x_mask = torch.ones((1, 1, 1, input_time_stamps) ).to(device)
                            # y_mask = subsequent_mask(input_time_stamps).to(device)
                            # out = model( x.permute(0,2,1).to(device), y.permute(0,2,1).to(device),x_mask, y_mask, x_t[:, 0, :].to(device), y_t[:, 0, :].to(device), 100, 100 ).to(device)
                            # out = out.permute(0,2,1)

                        elif args.dataset_name == "E2":
                            out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


                if norm:
                    p.print(f"doing normalisation inverse: {norm}")
                    out = normalizer.inverse(out)

                #test_print_time(args,b, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, input_time_stamps )

                if args.predict_difference:
                    if args.dataset_name == "KS1":
                        out = x + (1/0.3)*out
                    else:
                        out = x + out


                if t == 0:
                    pred = out.clone()
                else:
                    pred = torch.cat((pred, out.clone()), -1)


                if output_time_stamps > input_time_stamps:
                    x = out[...,-input_time_stamps:]
                    x_t = y_t[...,-input_time_stamps:]
                    x_tindicies = y_tindicies[...,-input_time_stamps:]

                elif output_time_stamps == input_time_stamps:
                    x = out #torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                    x_t = y_t #torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                    x_tindicies = y_tindicies #torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                
                elif output_time_stamps < input_time_stamps:
                    x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
                    x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                    x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                
                # x = torch.cat((x[..., next_input_time_stamps:], out[...,:next_input_time_stamps]), dim=-1)
                # x_t = torch.cat((x_t[..., next_input_time_stamps:], y_t[..., :next_input_time_stamps]), dim=-1)
                # x_tindicies = torch.cat((x_tindicies[..., next_input_time_stamps:], y_tindicies[...,:next_input_time_stamps]), dim=-1)
                    
            if b == 0:
                train_prediction = pred.clone()
            else:
                train_prediction = torch.cat((train_prediction,pred), dim=0)

    #test_l2_full = torch.mean((train_prediction-train_actual)**2, dim=[0,1] ).sum()
    #print("train_prediction, train_actual -->", train_prediction.shape, train_actual.shape)
    #test_l2_full = torch.mean((train_prediction[...,:train_actual.shape[-1]] - train_actual)**2 )
    #p.print(f"train_actual: {train_actual.shape}")
    test_l2_full = normalized_rmse(train_prediction[...,:train_actual.shape[-1]], train_actual)
    result = [test_l2_full, train_prediction[...,:train_actual.shape[-1]], train_actual, xy_tindicies]
    #p.print(f"train_actual: {train_actual.shape}")
    return result






def constant_one_to_one_test( args, model, loader, timestamps, dt_step, t_resolution, norm ):
    model.eval()
    #next_input_time_stamps = args.next_input_time_stamps
    #t_resolution = args.t_resolution
    #data_batchsize = args.batch_size_test
    input_time_stamps = args.input_time_stamps
    output_time_stamps = args.output_time_stamps
    time_sampling_choice = int(1)
    t_sample_space = torch.arange(t_resolution).to(device)
    tsamples = len(t_sample_space[::dt_step])

    #horizon = int( (tsamples-output_time_stamps)/output_time_stamps)
    if (tsamples-input_time_stamps)%output_time_stamps  == 0:
        horizon = int((tsamples-input_time_stamps)/output_time_stamps)
    else:
        horizon = int((tsamples-input_time_stamps)/output_time_stamps) + 1

    #horizon = round( (tsamples-input_time_stamps)/output_time_stamps) + 1

    #import pdb; pdb.set_trace()
    #p.print(f"horizon_test: {horizon}")
    with torch.no_grad():
        

        for b, (data, u_super, x, parameters) in enumerate(loader):

            data = data.to(device)
            current_batch_size = data.size(0)
            parameters = parameters[...,:args.no_parameters].to(device)
            
            data_batch = test_batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(current_batch_size, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
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
            time_stamps = [i for i in range(0, time_indicies.shape[-1]+output_time_stamps, output_time_stamps)]
            
            # x = xy[..., :input_time_stamps ]
            # x_t = xy_t[..., :input_time_stamps ]
            # x_tindicies = xy_tindicies[..., :input_time_stamps ]

            if b == 0:
                train_actual = xy[..., input_time_stamps:].clone()
            else:
                train_actual = torch.cat((train_actual, xy[..., input_time_stamps:]), dim=0)

            # input:
            x = xy[..., : input_time_stamps]
            x_t = xy_t[..., :input_time_stamps]
            x_tindicies = xy_tindicies[..., :input_time_stamps ]
            

            for t in range(horizon):

                # y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                # y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                # y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

                y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]


                if norm:
                    x = normalizer(x)  


                if args.time_prediction == "constant":
                    if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                        out = model(x).to(device)
                    elif args.dataset_name == "E2":
                        out = model(torch.cat((x, parameters), dim=-1)).to(device)
                    elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
                        out = model(torch.cat((x, parameters), dim=-1)).to(device)


                if args.time_prediction == "variable":

                    if args.time_conditioning == "addition":
                        x_x_t = x + x_t
                        if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                            out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)

                        elif args.dataset_name == "E2":
                            out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)


                    elif args.time_conditioning == "concatenate":
                        if args.dataset_name == "E1" or "B1" or "A1":

                            if y_t.shape[-1] != output_time_stamps:
                                y_t = y_t.repeat(1,1,output_time_stamps)

                            out = model( x, x_t, y_t ).to(device)

                        elif args.dataset_name == "E2":
                            out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                    
                    
                    elif args.time_conditioning == "attention":
                        if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                            #out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                            #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")
                            if y_t.shape[-1] != output_time_stamps:
                                y_t = y_t.repeat(1,1,output_time_stamps)
                                #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")

                            out = model( x, x_t, y_t ).to(device)
                            # x_mask = torch.ones((1, 1, 1, input_time_stamps) ).to(device)
                            # y_mask = subsequent_mask(input_time_stamps).to(device)
                            # out = model( x.permute(0,2,1).to(device), y.permute(0,2,1).to(device),x_mask, y_mask, x_t[:, 0, :].to(device), y_t[:, 0, :].to(device), 100, 100 ).to(device)
                            # out = out.permute(0,2,1)

                        elif args.dataset_name == "E2":
                            out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


                if norm:
                    out = normalizer.inverse(out)


                #predict_difference = True
                if args.predict_difference:
                    
                    if args.dataset_name == "KS1":
                        out = x + (1/0.3)*out
                    else:
                        out = x + out


                #test_print_time(args,b, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, input_time_stamps )


                if t == 0:
                    pred = out.clone()
                else:
                    pred = torch.cat((pred, out.clone()), -1)


                #import pdb; pdb.set_trace()
                if output_time_stamps > input_time_stamps:
                    x = y[...,-input_time_stamps:]
                    x_t = y_t[...,-input_time_stamps:]
                    x_tindicies = y_tindicies[...,-input_time_stamps:]

                elif output_time_stamps == input_time_stamps:
                    x = y #torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                    x_t = y_t #torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                    x_tindicies = y_tindicies #torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                
                elif output_time_stamps < input_time_stamps:
                    x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], y), dim=-1)
                    x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                    x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                
                # x = torch.cat((x[..., next_input_time_stamps:], y[...,:next_input_time_stamps]), dim=-1)
                # x_t = torch.cat((x_t[..., next_input_time_stamps:], y_t[..., :next_input_time_stamps]), dim=-1)
                # x_tindicies = torch.cat((x_tindicies[..., next_input_time_stamps:], y_tindicies[...,:next_input_time_stamps]), dim=-1)
                

                
            if b == 0:
                train_prediction = pred.clone()
            else:
                train_prediction = torch.cat((train_prediction,pred), dim=0)

    #test_l2_full = torch.mean((train_prediction-train_actual)**2, dim=[0,1] ).sum()
    #test_l2_full = torch.mean((train_prediction-train_actual)**2)
    #test_l2_full = torch.mean((train_prediction[...,:train_actual.shape[-1]] - train_actual)**2 )
    
    test_l2_full = normalized_rmse(train_prediction[...,:train_actual.shape[-1]], train_actual)
    result = [test_l2_full, train_prediction, train_actual, xy_tindicies]
    return result








# def constant_one_step_test( args, model, loader, timestamps, dt_step, t_resolution, norm ):
#     model.eval()
#     #next_input_time_stamps = args.next_input_time_stamps
#     #t_resolution = args.t_resolution
#     #data_batchsize = args.batch_size_test
#     input_time_stamps = args.input_time_stamps
#     output_time_stamps = args.output_time_stamps
#     time_sampling_choice = int(1)
#     t_sample_space = torch.arange(t_resolution).to(device)
#     tsamples = len(t_sample_space[::dt_step])

#     #horizon = int( (tsamples-output_time_stamps)/output_time_stamps)
#     if (tsamples-input_time_stamps)%output_time_stamps  == 0:
#         horizon = int((tsamples-input_time_stamps)/output_time_stamps)
#     else:
#         horizon = int((tsamples-input_time_stamps)/output_time_stamps) + 1

#     #horizon = round( (tsamples-input_time_stamps)/output_time_stamps) + 1

#     #import pdb; pdb.set_trace()
#     # p.print("Testing...............")
#     # p.print(f"horizon_test: {horizon}")
#     with torch.no_grad():
        

#         for b, (data, u_super, x, parameters) in enumerate(loader):

#             data = data.to(device)
#             current_batch_size = data.size(0)
#             parameters = parameters[...,:args.no_parameters].to(device)

#             size_x = data.shape[1]
#             batch_size = data.shape[0]
#             grid = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).reshape(1, size_x, 1).repeat([batch_size, 1, 1]).to(device)

#             data_batch = test_batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(current_batch_size, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
#             time_indicies = t_sample_space[data_batch.indicies]
#             xy = torch.gather(data, -1, time_indicies.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
#             xy_t = torch.ones_like(xy)[:,0,:].to(device)
#             xy_t = xy_t*timestamps[time_indicies]


#             # p.print(f"xy_t: {xy_t.shape}")
#             # p.print(f"xy_t: {xy_t[:3,:10]}")


#             ###########################################
#             ######################
#             # Time Difference
#             #xy_t = torch.cat((torch.diff(xy_t, dim=-1), torch.zeros(xy_t.shape[0], 1).to(device)), dim=-1)
#             ###########################################
#             ##########################  
#             # p.print(f"xy_t: {xy_t.shape}")
#             # p.print(f"xy_t: {xy_t[:3,:10]}")


#             xy_t = xy_t.unsqueeze(1).repeat(1,data.shape[1],1)
#             xy_tindicies = time_indicies.long()
#             time_stamps = [i for i in range(0, time_indicies.shape[-1]+output_time_stamps, output_time_stamps)]
            
#             # x = xy[..., :input_time_stamps ]
#             # x_t = xy_t[..., :input_time_stamps ]
#             # x_tindicies = xy_tindicies[..., :input_time_stamps ]

#             if b == 0:
#                 train_actual = xy[..., input_time_stamps:].clone()
#             else:
#                 train_actual = torch.cat((train_actual, xy[..., input_time_stamps:]), dim=0)

#             # input:
#             x = xy[..., : input_time_stamps]
#             x_t = xy_t[..., :input_time_stamps]
#             x_tindicies = xy_tindicies[..., :input_time_stamps ]
            

#             for t in range(horizon):

#                 # y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
#                 # y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
#                 # y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

#                 y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
#                 y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
#                 y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]

#                 # p.print(f"x_t: {x_t.shape}")
#                 # p.print(f"x_t: {x_t[:5,1,::4]}")

#                 # p.print(f"y_t: {y_t.shape}")
#                 # p.print(f"y_t: {y_t[:5,1,::4]}")

#                 y_t = y_t - x_t

#                 # p.print(f"y_t: {y_t.shape}")
#                 # p.print(f"y_t: {y_t[:5,1,::4]}")

                
#                 # if norm:
#                 #     x = normalizer(x)  


#                 if args.time_prediction == "constant":
#                     if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
#                         #print(x.shape, grid.shape, y_t.shape)
#                         out = model(x, grid, grid[:,0,:], y_t[:,0,:]).squeeze(-1).to(device)
#                         #print(out.shape)
#                     elif args.dataset_name == "E2":
#                         out = model(torch.cat((x, parameters), dim=-1)).to(device)
#                     elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
#                         out = model(torch.cat((x, parameters), dim=-1)).to(device)


#                 if args.time_prediction == "variable":

#                     if args.time_conditioning == "addition":
#                         x_x_t = x + x_t
#                         if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
#                             out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)

#                         elif args.dataset_name == "E2":
#                             out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)


#                     elif args.time_conditioning == "concatenate":
#                         if args.dataset_name == "E1" or "B1" or "A1":

#                             if y_t.shape[-1] != output_time_stamps:
#                                 y_t = y_t.repeat(1,1,output_time_stamps)

#                             out = model( x, x_t, y_t ).to(device)

#                         elif args.dataset_name == "E2":
#                             out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                    
                    
#                     elif args.time_conditioning == "attention":
#                         if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
#                             #out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
#                             #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")
#                             if y_t.shape[-1] != output_time_stamps:
#                                 y_t = y_t.repeat(1,1,output_time_stamps)
#                                 #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")

#                             out = model( x, x_t, y_t ).to(device)
#                             # x_mask = torch.ones((1, 1, 1, input_time_stamps) ).to(device)
#                             # y_mask = subsequent_mask(input_time_stamps).to(device)
#                             # out = model( x.permute(0,2,1).to(device), y.permute(0,2,1).to(device),x_mask, y_mask, x_t[:, 0, :].to(device), y_t[:, 0, :].to(device), 100, 100 ).to(device)
#                             # out = out.permute(0,2,1)

#                         elif args.dataset_name == "E2":
#                             out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


#                 # if norm:
#                 #     out = normalizer.inverse(out)


#                 #predict_difference = True
#                 # if args.predict_difference:
                    
#                 #     if args.dataset_name == "KS1":
#                 #         out = x + (1/0.3)*out
#                 #     else:
#                 #         out = x + out


#                 #test_print_time(args,b, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, input_time_stamps )


#                 if t == 0:
#                     pred = out.clone()
#                 else:
#                     pred = torch.cat((pred, out.clone()), -1)


#                 #import pdb; pdb.set_trace()
#                 if output_time_stamps > input_time_stamps:
#                     x = y[...,-input_time_stamps:]
#                     x_t = y_t[...,-input_time_stamps:]
#                     x_tindicies = y_tindicies[...,-input_time_stamps:]

#                 elif output_time_stamps == input_time_stamps:
#                     x = y #torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
#                     x_t = y_t #torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
#                     x_tindicies = y_tindicies #torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                
#                 elif output_time_stamps < input_time_stamps:
#                     x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], y), dim=-1)
#                     x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
#                     x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                
#                 # x = torch.cat((x[..., next_input_time_stamps:], y[...,:next_input_time_stamps]), dim=-1)
#                 # x_t = torch.cat((x_t[..., next_input_time_stamps:], y_t[..., :next_input_time_stamps]), dim=-1)
#                 # x_tindicies = torch.cat((x_tindicies[..., next_input_time_stamps:], y_tindicies[...,:next_input_time_stamps]), dim=-1)
                

                
#             if b == 0:
#                 train_prediction = pred.clone()
#             else:
#                 train_prediction = torch.cat((train_prediction,pred), dim=0)

#     #test_l2_full = torch.mean((train_prediction-train_actual)**2, dim=[0,1] ).sum()
#     #test_l2_full = torch.mean((train_prediction-train_actual)**2)
#     #test_l2_full = torch.mean((train_prediction[...,:train_actual.shape[-1]] - train_actual)**2 )
    
#     test_l2_full = normalized_rmse(train_prediction[...,:train_actual.shape[-1]], train_actual)
#     result = [test_l2_full, train_prediction, train_actual, xy_tindicies]
#     return result










def constant_one_step_test( args, model, loader, timestamps, dt_step, t_resolution, norm ):
    model.eval()
    #next_input_time_stamps = args.next_input_time_stamps
    #t_resolution = args.t_resolution
    #data_batchsize = args.batch_size_test
    input_time_stamps = args.input_time_stamps
    output_time_stamps = args.output_time_stamps
    time_sampling_choice = int(1)
    t_sample_space = torch.arange(t_resolution).to(device)
    tsamples = len(t_sample_space[::dt_step])

    #horizon = int( (tsamples-output_time_stamps)/output_time_stamps)
    # if (tsamples-input_time_stamps)%output_time_stamps  == 0:
    #     horizon = int((tsamples-input_time_stamps)/output_time_stamps)
    # else:
    #     horizon = int((tsamples-input_time_stamps)/output_time_stamps) + 1

    #horizon = round( (tsamples-input_time_stamps)/output_time_stamps) + 1
    initial_step = input_time_stamps
    t_train = output_time_stamps + 1
    criterion = torch.nn.MSELoss(reduction="none")
    criterion_1 = torch.nn.MSELoss(reduction="mean")
    #import pdb; pdb.set_trace()
    #p.print("Testing...............")
    # p.print(f"horizon_test: {horizon}")
    val_l2_step = 0
    val_l2_full = 0
    val_l2_full_mean = 0

    with torch.no_grad():

        for b, (x, u_super, yyz, pde_param) in enumerate(loader):

            loss = 0
            xx = x[...,::dt_step][..., :initial_step].to(device)
            yy = x[...,::dt_step][..., initial_step:t_train].to(device)

            size_x = xx.shape[1]
            batch_size = xx.shape[0]
            grid = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).reshape(1, size_x, 1).repeat([batch_size, 1, 1]).to(device)

            pde_param = pde_param.to(device)

            #yy = yy[..., 0:t_train, :]

            # Prepare queried times t in [0..1]
            t = torch.arange(initial_step, t_train, device=xx.device) * 1 / (t_train-1)
            t = t.repeat((xx.size(0), 1))
            # Prepare queried times t in [0..1]
            #t = torch.arange(initial_step, yy.shape[-2], device=xx.device) * 1 / (t_train-1)
            #t = t.repeat((xx.size(0), 1))

            # Forward pass
            #print("input: ", xx.shape, grid.shape, pde_param.shape, t.shape)
            pred = model(xx, grid, pde_param, t)
            #print("output: ", pred.shape)
            #pred = torch.cat((xx, pred), dim=-2)

            # Loss calculation
            _batch = yy.size(0)
            #print("pred and y: ", pred.squeeze(-1).shape, yy.shape)
            loss = torch.sum(torch.mean(criterion(pred.squeeze(-1), yy), dim=(0, 1)))
            #print("loss: ", loss)
            #print("pred and y reshape: ", pred.reshape(_batch, -1).shape, yy.reshape(_batch, -1).shape)
            l2_full = criterion_1(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()
            #print("l2_full: ", l2_full)

            val_l2_step += loss.item()
            val_l2_full += l2_full
            val_l2_full_mean += l2_full * _batch

        # Calculate mean of l2 full loss
        val_l2_full_mean = val_l2_full_mean / len(loader)


                # if norm:
                #     out = normalizer.inverse(out)


                #predict_difference = True
                # if args.predict_difference:
                    
                #     if args.dataset_name == "KS1":
                #         out = x + (1/0.3)*out
                #     else:
                #         out = x + out


                #test_print_time(args,b, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, input_time_stamps )

    xy_tindicies = 0
    test_l2_full = normalized_rmse(pred.squeeze(-1), yy)
    result = [test_l2_full, pred, yy, xy_tindicies]
    return result






















def variable_rollout_test( args, model, loader, timestamps, dt_step, t_resolution, norm, no_of_steps=None):

    model.eval()
    #input_time_stamps = args.input_time_stamps
    #t_resolution = args.t_resolution
    data_batchsize = args.batch_size_train
    input_time_stamps = args.input_time_stamps
    output_time_stamps = args.output_time_stamps

    #time_sampling_choice = int(3)
    if args.mode == "train":
        time_sampling_choice = int(2)
    elif args.mode == "test":
        time_sampling_choice = int(3)

    #t_sample_space = torch.arange(t_resolution).to(device)
    # p.print(f"t_resolution: {t_resolution}")
    # p.print(f"timestamps: {timestamps.shape}")
    t_sample_space = torch.arange(t_resolution)[::int(dt_step)].to(device)

    #timestamps = torch.tensor(timestamps).to(device)
    # if no_of_steps == None:
    #     tsamples = len(t_sample_space[::int(dt_step)])
    # else:
    #     tsamples = no_of_steps


    if no_of_steps == None:
        tsamples = len(t_sample_space)
    else:
        assert no_of_steps <=  len(t_sample_space)
        tsamples = int(no_of_steps)

    #horizon = int((tsamples-input_time_stamps)/output_time_stamps)
    if (tsamples-input_time_stamps)%output_time_stamps  == 0:
        horizon = int((tsamples-input_time_stamps)/output_time_stamps)
    else:
        horizon = int((tsamples-input_time_stamps)/output_time_stamps) + 1

    #import pdb; pdb.set_trace()
    #p.print(f"horizon_test: {horizon}")


    with torch.no_grad():
        

        for b, (data, u_super, x, parameters) in enumerate(loader):

            data = data.to(device)
            parameters = parameters[...,:args.no_parameters].to(device)
            
            #data_batch = test_batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
            data_batch = test_batch_time_sampling(choice=time_sampling_choice, total_range = len(t_sample_space),  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
            time_indicies = t_sample_space[data_batch.indicies]

            # p.print(f"data: {data.shape}")
            # p.print(f"time_indicies: {time_indicies.shape}")
            #print("\n")
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
            time_stamps = [i for i in range(0, time_indicies.shape[-1]+output_time_stamps, output_time_stamps)]
            
            # print("horizon", horizon)
            # print("time_indicies", time_indicies.shape)
            # print("time_stamps", time_stamps)
            # x = xy[..., :output_time_stamps ]
            # x_t = xy_t[..., :output_time_stamps ]
            # x_tindicies = xy_tindicies[..., :output_time_stamps ]

            # x = xy[..., :input_time_stamps ]
            # x_t = xy_t[..., :input_time_stamps ]
            # x_tindicies = xy_tindicies[..., :input_time_stamps ]
            #import pdb; pdb.set_trace()


            if b == 0:
                train_actual = xy[..., input_time_stamps:].clone()
            else:
                train_actual = torch.cat((train_actual, xy[..., input_time_stamps:]), dim=0)

            # input:
            x = xy[..., : input_time_stamps]
            x_t = xy_t[..., :input_time_stamps]
            x_tindicies = xy_tindicies[..., :input_time_stamps ]
            
            #print("xy -->", xy.shape)
            #import pdb; pdb.set_trace()
            for t in range(horizon):
                #p.print(f"t: {t}")
                #import pdb; pdb.set_trace()
                #import pdb; pdb.set_trace()


                # y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                # y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                # y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]
                #print("in_range ->", )
                # print("t ->",t)
                # print("out_range ->", input_time_stamps+time_stamps[t],input_time_stamps+time_stamps[t+1])
                # print("\n")
                y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]

                #print("args.norm -->", args.norm)

                if norm:
                    p.print(f"doing normalisation: {norm}")
                    x = normalizer(x)  


                if args.time_prediction == "constant":
                    if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                        out = model(x).to(device)
                    elif args.dataset_name == "E2":
                        out = model(torch.cat((x, parameters), dim=-1)).to(device)
                    elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
                        out = model(torch.cat((x, parameters), dim=-1)).to(device)


                if args.time_prediction == "variable":

                    if args.time_conditioning == "addition":
                        x_x_t = x + x_t
                        if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                            out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                        elif args.dataset_name == "E2":
                            out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                    elif args.time_conditioning == "concatenate":
                        if args.dataset_name == "E1" or "B1" or "A1":

                            if y_t.shape[-1] != output_time_stamps:
                                y_t = y_t.repeat(1,1,output_time_stamps)

                            out = model( x, x_t, y_t ).to(device)

                        elif args.dataset_name == "E2":
                            out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                    
                    
                    elif args.time_conditioning == "attention":
                        if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":

                            #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")
                            if y_t.shape[-1] != output_time_stamps:
                                #y_t = y_t.repeat(1,1,output_time_stamps)
                                remain_step = int(output_time_stamps/y_t.shape[-1]) + 1
                                y_t = y_t.repeat(1,1,remain_step*y_t.shape[-1])[...,:output_time_stamps]
                                #y_t = torch.cat((y_t, y_t[..., :remain_step]), dim=-1)
                                #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")

                            # p.print(f"x_t: {x_t[:5,0,]}")
                            # p.print(f"y_t: {y_t[:5,0,]}")
                            #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")
                            out = model( x, x_t, y_t ).to(device)
                            #import pdb; pdb.set_trace()
                            #out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                            # x_mask = torch.ones((1, 1, 1, input_time_stamps) ).to(device)
                            # y_mask = subsequent_mask(input_time_stamps).to(device)
                            # out = model( x.permute(0,2,1).to(device), y.permute(0,2,1).to(device),x_mask, y_mask, x_t[:, 0, :].to(device), y_t[:, 0, :].to(device), 100, 100 ).to(device)
                            # out = out.permute(0,2,1)

                        elif args.dataset_name == "E2":
                            out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


                if norm:
                    p.print(f"doing normalisation inverse: {norm}")
                    out = normalizer.inverse(out)

                #test_print_time(args,b, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, input_time_stamps )

                if args.predict_difference:
                    if args.dataset_name == "KS1":
                        out = x + (1/0.3)*out
                    else:
                        out = x + out


                if t == 0:
                    pred = out.clone()
                else:
                    pred = torch.cat((pred, out.clone()), -1)


                if output_time_stamps > input_time_stamps:
                    x = out[...,-input_time_stamps:]
                    x_t = y_t[...,-input_time_stamps:]
                    x_tindicies = y_tindicies[...,-input_time_stamps:]

                elif output_time_stamps == input_time_stamps:
                    x = out #torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                    x_t = y_t #torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                    x_tindicies = y_tindicies #torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                
                elif output_time_stamps < input_time_stamps:
                    x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], out), dim=-1)
                    x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                    x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                
                # x = torch.cat((x[..., next_input_time_stamps:], out[...,:next_input_time_stamps]), dim=-1)
                # x_t = torch.cat((x_t[..., next_input_time_stamps:], y_t[..., :next_input_time_stamps]), dim=-1)
                # x_tindicies = torch.cat((x_tindicies[..., next_input_time_stamps:], y_tindicies[...,:next_input_time_stamps]), dim=-1)
                    
            if b == 0:
                train_prediction = pred.clone()
            else:
                train_prediction = torch.cat((train_prediction,pred), dim=0)

    #test_l2_full = torch.mean((train_prediction-train_actual)**2, dim=[0,1] ).sum()
    #print("train_prediction, train_actual -->", train_prediction.shape, train_actual.shape)
    
    test_l2_full = normalized_rmse(train_prediction[...,:train_actual.shape[-1]], train_actual)
    #test_l2_full = torch.mean((train_prediction[...,:train_actual.shape[-1]] - train_actual)**2 )

    result = [test_l2_full, train_prediction[...,:train_actual.shape[-1]], train_actual, xy_tindicies]
    return result






def variable_one_to_one_test( args, model, loader, timestamps, dt_step, t_resolution, norm, no_of_steps=None ):
    model.eval()
    #next_input_time_stamps = args.next_input_time_stamps
    #t_resolution = args.t_resolution
    data_batchsize = args.batch_size_train
    input_time_stamps = args.input_time_stamps
    output_time_stamps = args.output_time_stamps

    if args.mode == "train":
        time_sampling_choice = int(2)
    elif args.mode == "test":
        time_sampling_choice = int(3)

    #t_sample_space = torch.arange(t_resolution).to(device)
    t_sample_space = torch.arange(t_resolution)[::int(dt_step)].to(device)
    #tsamples = len(t_sample_space[::dt_step])

    if no_of_steps == None:
        tsamples = len(t_sample_space)
    else:
        assert no_of_steps <=  len(t_sample_space)
        tsamples = int(no_of_steps)

    #horizon = int( (tsamples-output_time_stamps)/output_time_stamps)
    if (tsamples-input_time_stamps)%output_time_stamps  == 0:
        horizon = int((tsamples-input_time_stamps)/output_time_stamps)
    else:
        horizon = int((tsamples-input_time_stamps)/output_time_stamps) + 1

    #horizon = round( (tsamples-input_time_stamps)/output_time_stamps) + 1

    #import pdb; pdb.set_trace()
    
    #p.print(f"horizon_test: {horizon}")
    with torch.no_grad():
        

        for b, (data, u_super, x, parameters) in enumerate(loader):

            data = data.to(device)
            parameters = parameters[...,:args.no_parameters].to(device)
            
            data_batch = test_batch_time_sampling(choice=time_sampling_choice, total_range = len(t_sample_space),  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
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
            time_stamps = [i for i in range(0, time_indicies.shape[-1]+output_time_stamps, output_time_stamps)]
            
            # x = xy[..., :input_time_stamps ]
            # x_t = xy_t[..., :input_time_stamps ]
            # x_tindicies = xy_tindicies[..., :input_time_stamps ]

            if b == 0:
                train_actual = xy[..., input_time_stamps:].clone()
            else:
                train_actual = torch.cat((train_actual, xy[..., input_time_stamps:]), dim=0)

            # input:
            x = xy[..., : input_time_stamps]
            x_t = xy_t[..., :input_time_stamps]
            x_tindicies = xy_tindicies[..., :input_time_stamps ]
            

            for t in range(horizon):

                # y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                # y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                # y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

                y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]


                if norm:
                    x = normalizer(x)  


                if args.time_prediction == "constant":
                    if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                        out = model(x).to(device)
                    elif args.dataset_name == "E2":
                        out = model(torch.cat((x, parameters), dim=-1)).to(device)
                    elif args.dataset_name == "KS1" or args.dataset_name == "KdV":
                        out = model(torch.cat((x, parameters), dim=-1)).to(device)


                if args.time_prediction == "variable":

                    if args.time_conditioning == "addition":
                        x_x_t = x + x_t
                        if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                            out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                        elif args.dataset_name == "E2":
                            out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                    elif args.time_conditioning == "concatenate":
                        if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":

                            if y_t.shape[-1] != output_time_stamps:
                                y_t = y_t.repeat(1,1,output_time_stamps)

                            out = model( x, x_t, y_t ).to(device)
                            
                        elif args.dataset_name == "E2":
                            out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                    
                    
                    elif args.time_conditioning == "attention":
                        if args.dataset_name == "E1" or args.dataset_name =="B1"  or args.dataset_name == "A1":
                            #out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)

                            # #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")
                            # if y_t.shape[-1] != output_time_stamps:
                            #     #y_t = y_t.repeat(1,1,output_time_stamps)
                            #     remain_step = output_time_stamps-y_t.shape[-1]
                            #     y_t = torch.cat((y_t, y_t[..., :remain_step]))
                            #     #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")

                            #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")
                            if y_t.shape[-1] != output_time_stamps:
                                #y_t = y_t.repeat(1,1,output_time_stamps)
                                remain_step = int(output_time_stamps/y_t.shape[-1]) + 1
                                y_t = y_t.repeat(1,1,remain_step*y_t.shape[-1])[...,:output_time_stamps]
                                #y_t = torch.cat((y_t, y_t[..., :remain_step]), dim=-1)
                                #p.print(f"x, x_t, y_t: {x.shape, x_t.shape, y_t.shape}")


                            out = model( x, x_t, y_t ).to(device)
                            # x_mask = torch.ones((1, 1, 1, input_time_stamps) ).to(device)
                            # y_mask = subsequent_mask(input_time_stamps).to(device)
                            # out = model( x.permute(0,2,1).to(device), y.permute(0,2,1).to(device),x_mask, y_mask, x_t[:, 0, :].to(device), y_t[:, 0, :].to(device), 100, 100 ).to(device)
                            # out = out.permute(0,2,1)

                        elif args.dataset_name == "E2":
                            out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


                if norm:
                    out = normalizer.inverse(out)


                #predict_difference = True
                if args.predict_difference:
                    
                    if args.dataset_name == "KS1":
                        out = x + (1/0.3)*out
                    else:
                        out = x + out


                #test_print_time(args,b, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, input_time_stamps )


                if t == 0:
                    pred = out.clone()
                else:
                    pred = torch.cat((pred, out.clone()), -1)


                #import pdb; pdb.set_trace()
                if output_time_stamps > input_time_stamps:
                    x = y[...,-input_time_stamps:]
                    x_t = y_t[...,-input_time_stamps:]
                    x_tindicies = y_tindicies[...,-input_time_stamps:]

                elif output_time_stamps == input_time_stamps:
                    x = y #torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                    x_t = y_t #torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                    x_tindicies = y_tindicies #torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                
                elif output_time_stamps < input_time_stamps:
                    x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], y), dim=-1)
                    x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                    x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                
                # x = torch.cat((x[..., next_input_time_stamps:], y[...,:next_input_time_stamps]), dim=-1)
                # x_t = torch.cat((x_t[..., next_input_time_stamps:], y_t[..., :next_input_time_stamps]), dim=-1)
                # x_tindicies = torch.cat((x_tindicies[..., next_input_time_stamps:], y_tindicies[...,:next_input_time_stamps]), dim=-1)
                

                
            if b == 0:
                train_prediction = pred.clone()
            else:
                train_prediction = torch.cat((train_prediction,pred), dim=0)

    #test_l2_full = torch.mean((train_prediction-train_actual)**2, dim=[0,1] ).sum()
    #test_l2_full = torch.mean((train_prediction-train_actual)**2)
    #test_l2_full = torch.mean((train_prediction[...,:train_actual.shape[-1]] - train_actual)**2 )
    test_l2_full = normalized_rmse(train_prediction[...,:train_actual.shape[-1]], train_actual)
    result = [test_l2_full, train_prediction, train_actual, xy_tindicies]
    return result


# def variable_rollout_test( args, model, loader, timestamps, tsamples, time_sampling_choice = 2 ):
#     model.eval()
#     assert time_sampling_choice >= 2
#     next_input_time_stamps = args.next_input_time_stamps
#     t_resolution = args.t_resolution
#     data_batchsize = args.batch_size_train
#     input_time_stamps = args.input_time_stamps
#     output_time_stamps = args.output_time_stamps
#     t_sample_space = torch.arange(t_resolution).to(device)
#     horizon = int( (tsamples-output_time_stamps)/output_time_stamps)


#     with torch.no_grad():
#         for b, (data, u_super, x, parameters) in enumerate(loader):

#             data = data.to(device)
#             parameters = parameters[...,:args.no_parameters].to(device)
            
#             data_batch = test_batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps)
#             time_indicies = t_sample_space[data_batch.indicies]
#             xy = torch.gather(data, -1, time_indicies.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
#             xy_t = torch.ones_like(xy)[:,0,:].to(device)
#             xy_t = xy_t*timestamps[time_indicies]
#             xy_t = xy_t.unsqueeze(1).repeat(1,data.shape[1],1)
#             xy_tindicies = time_indicies.long()
#             time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, next_input_time_stamps)]
            
#             x = xy[..., :output_time_stamps ]
#             x_t = xy_t[..., :output_time_stamps ]
#             x_tindicies = xy_tindicies[..., :output_time_stamps ]

#             if b == 0:
#                 train_actual = xy[..., input_time_stamps:].clone()
#             else:
#                 train_actual = torch.cat((train_actual, xy[..., input_time_stamps:]), dim=0)


#             x = xy[..., : input_time_stamps]
#             x_t = xy_t[..., :input_time_stamps]
#             x_tindicies = xy_tindicies[..., :input_time_stamps ]
            

#             for t in range(horizon):
#                 y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
#                 y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
#                 y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

#                 if args.time_prediction.startswith("constant"):
#                     if args.dataset_name.endswith("E1"):
#                         out = model(x).to(device)
#                     elif args.dataset_name.endswith("E2"):
#                         out = model(torch.cat((x, parameters), dim=-1)).to(device)


#                 if args.time_prediction.startswith("variable"):

#                     if args.time_conditioning.startswith("addition"):
#                         x_x_t = x + x_t
#                         if args.dataset_name.endswith("E1"):
#                             out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
#                         elif args.dataset_name.endswith("E2"):
#                             out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

#                     elif args.time_conditioning.startswith("concatenate"):
#                         if args.dataset_name.endswith("E1"):
#                             out = model( x, x_t, y_t ).to(device)
#                         elif args.dataset_name.endswith("E2"):
#                             out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                    
                    
#                     elif args.time_conditioning.startswith("attention"):
#                         if args.dataset_name.endswith("E1"):
#                             out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
#                         elif args.dataset_name.endswith("E2"):
#                             out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)



#                 #test_print_time(args,b, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies )



#                 if t == 0:
#                     pred = out[...,:next_input_time_stamps]
#                 else:
#                     pred = torch.cat((pred, out[...,:next_input_time_stamps]), -1)


#                 x = torch.cat((x[..., next_input_time_stamps:], out[...,:next_input_time_stamps]), dim=-1)
#                 x_t = torch.cat((x_t[..., next_input_time_stamps:], y_t[..., :next_input_time_stamps]), dim=-1)
#                 x_tindicies = torch.cat((x_tindicies[..., next_input_time_stamps:], y_tindicies[...,:next_input_time_stamps]), dim=-1)

                    
#             if b == 0:
#                 train_prediction = pred.clone()
#             else:
#                 train_prediction = torch.cat((train_prediction,pred), dim=0)

#     #test_l2_full = torch.mean((train_prediction-train_actual)**2, dim=[0,1] ).sum()
#     test_l2_full = torch.mean((train_prediction-train_actual)**2 )

#     result = [test_l2_full, train_prediction, train_actual, xy_tindicies]
#     return result




# def variable_one_to_one_test( args, model, loader, timestamps, tsamples, time_sampling_choice = 2 ):
#     model.eval()
#     assert time_sampling_choice >= 2
#     next_input_time_stamps = args.next_input_time_stamps
#     t_resolution = args.t_resolution
#     data_batchsize = args.batch_size_train
#     input_time_stamps = args.input_time_stamps
#     output_time_stamps = args.output_time_stamps
#     t_sample_space = torch.arange(t_resolution).to(device)
#     horizon = int( (tsamples-output_time_stamps)/output_time_stamps)


#     with torch.no_grad():
#         for b, (data, u_super, x, parameters) in enumerate(loader):

#             data = data.to(device)
#             parameters = parameters[...,:args.no_parameters].to(device)
            
#             data_batch = test_batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps)
#             time_indicies = t_sample_space[data_batch.indicies]
#             xy = torch.gather(data, -1, time_indicies.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
#             xy_t = torch.ones_like(xy)[:,0,:].to(device)
#             xy_t = xy_t*timestamps[time_indicies]
#             xy_t = xy_t.unsqueeze(1).repeat(1,data.shape[1],1)
#             xy_tindicies = time_indicies.long()
#             time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, next_input_time_stamps)]
            
#             x = xy[..., :output_time_stamps ]
#             x_t = xy_t[..., :output_time_stamps ]
#             x_tindicies = xy_tindicies[..., :output_time_stamps ]

#             if b == 0:
#                 train_actual = xy[..., input_time_stamps:].clone()
#             else:
#                 train_actual = torch.cat((train_actual, xy[..., input_time_stamps:]), dim=0)


#             x = xy[..., : input_time_stamps]
#             x_t = xy_t[..., :input_time_stamps]
#             x_tindicies = xy_tindicies[..., :input_time_stamps ]
            

#             for t in range(horizon):
#                 y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
#                 y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
#                 y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

#                 if args.time_prediction.startswith("constant"):
#                     if args.dataset_name.endswith("E1"):
#                         out = model(x).to(device)
#                     elif args.dataset_name.endswith("E2"):
#                         out = model(torch.cat((x, parameters), dim=-1)).to(device)


#                 if args.time_prediction.startswith("variable"):

#                     if args.time_conditioning.startswith("addition"):
#                         x_x_t = x + x_t
#                         if args.dataset_name.endswith("E1"):
#                             out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
#                         elif args.dataset_name.endswith("E2"):
#                             out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

#                     elif args.time_conditioning.startswith("concatenate"):
#                         if args.dataset_name.endswith("E1"):
#                             out = model( x, x_t, y_t ).to(device)
#                         elif args.dataset_name.endswith("E2"):
#                             out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                    
                    
#                     elif args.time_conditioning.startswith("attention"):
#                         if args.dataset_name.endswith("E1"):
#                             out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
#                         elif args.dataset_name.endswith("E2"):
#                             out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)



#                 #test_print_time(args,b, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies )



#                 if t == 0:
#                     pred = out[...,:next_input_time_stamps]
#                 else:
#                     pred = torch.cat((pred, out[...,:next_input_time_stamps]), -1)


#                 x = torch.cat((x[..., next_input_time_stamps:], y[...,:next_input_time_stamps]), dim=-1)
#                 x_t = torch.cat((x_t[..., next_input_time_stamps:], y_t[..., :next_input_time_stamps]), dim=-1)
#                 x_tindicies = torch.cat((x_tindicies[..., next_input_time_stamps:], y_tindicies[...,:next_input_time_stamps]), dim=-1)

                    
#             if b == 0:
#                 train_prediction = pred.clone()
#             else:
#                 train_prediction = torch.cat((train_prediction,pred), dim=0)

#     #test_l2_full = torch.mean((train_prediction-train_actual)**2, dim=[0,1] ).sum()
#     test_l2_full = torch.mean((train_prediction-train_actual)**2 )

#     result = [test_l2_full, train_prediction, train_actual, xy_tindicies]
#     return result





def constant_one_to_one_test_splits( args, model, loader, timestamps, dt_step ):

    model.eval()
    #input_time_stamps = args.input_time_stamps
    t_resolution = args.t_resolution
    data_batchsize = args.batch_size_train
    input_time_stamps = args.input_time_stamps
    output_time_stamps = args.output_time_stamps
    time_sampling_choice = int(1)
    t_sample_space = torch.arange(t_resolution).to(device)

    #timestamps = torch.tensor(timestamps).to(device)
    tsamples = len(t_sample_space[::int(dt_step)])

    #horizon = int((tsamples-input_time_stamps)/output_time_stamps)
    horizon = round((tsamples-input_time_stamps)/output_time_stamps) + 1

    #import pdb; pdb.set_trace()

    with torch.no_grad():
        

        for b, (data, u_super, x, parameters) in enumerate(loader):

            data = data.to(device)
            parameters = parameters[...,:args.no_parameters].to(device)
            
            data_batch = test_batch_time_sampling(choice=time_sampling_choice, total_range = t_resolution,  no_of_samp=(data_batchsize, tsamples), t_pred_steps= output_time_stamps, dt=dt_step)
            time_indicies = t_sample_space[data_batch.indicies]
            xy = torch.gather(data, -1, time_indicies.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
            xy_t = torch.ones_like(xy)[:,0,:].to(device)
            xy_t = xy_t*timestamps[time_indicies]
            xy_t = xy_t.unsqueeze(1).repeat(1,data.shape[1],1)
            xy_tindicies = time_indicies.long()
            time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, output_time_stamps)]
            
            # x = xy[..., :output_time_stamps ]
            # x_t = xy_t[..., :output_time_stamps ]
            # x_tindicies = xy_tindicies[..., :output_time_stamps ]

            # x = xy[..., :input_time_stamps ]
            # x_t = xy_t[..., :input_time_stamps ]
            # x_tindicies = xy_tindicies[..., :input_time_stamps ]
            
            if b == 0:
                train_actual = xy_t[:, 0, :-input_time_stamps].clone()
            else:
                train_actual = torch.cat((train_actual,  xy_t[:, 0, :-input_time_stamps].clone()), dim=0)

            # input:
            x = xy[..., : input_time_stamps]
            x_t = xy_t[..., :input_time_stamps]
            x_tindicies = xy_tindicies[..., :input_time_stamps ]


            time_indicies_zero = torch.zeros( (data.shape[0], 1) ).to(torch.long)
            x0 = torch.gather(data, -1, time_indicies_zero.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )
            x0_t = torch.ones_like(x0)[:,0,:].to(device)
            x0_t = x0_t*timestamps[time_indicies_zero]
            x0_t = x0_t.unsqueeze(1).repeat(1,data.shape[1],1)
            xy_tindicies_zero = time_indicies_zero.long()



            for t in range(horizon):
                #import pdb; pdb.set_trace()


                # y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                # y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                # y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

                y = xy[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                y_t = xy_t[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]
                y_tindicies = xy_tindicies[..., input_time_stamps+time_stamps[t]:input_time_stamps+time_stamps[t+1]]

                #print("args.norm -->", args.norm)

                if args.norm:
                    x = normalizer(x)  


                if args.time_prediction.startswith("constant"):
                    if args.dataset_name.endswith("E1"):
                        out = model(torch.cat((x0, x), dim=-1)).to(device)
                    elif args.dataset_name.endswith("E2"):
                        out = model(torch.cat((x, parameters), dim=-1)).to(device)


                if args.time_prediction.startswith("variable"):

                    if args.time_conditioning.startswith("addition"):
                        x_x_t = x + x_t
                        if args.dataset_name.endswith("E1"):
                            out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
                        elif args.dataset_name.endswith("E2"):
                            out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

                    elif args.time_conditioning.startswith("concatenate"):
                        if args.dataset_name.endswith("E1"):
                            out = model( x, x_t, y_t ).to(device)
                        elif args.dataset_name.endswith("E2"):
                            out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)
                    
                    
                    elif args.time_conditioning.startswith("attention"):
                        if args.dataset_name.endswith("E1"):
                            out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
                        elif args.dataset_name.endswith("E2"):
                            out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


                if args.norm:
                    out = normalizer.inverse(out)

                #test_print_time(args,b, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, input_time_stamps )

                if t == 0:
                    pred = out.clone()
                else:
                    pred = torch.cat((pred, out.clone()), -1)


                if output_time_stamps > input_time_stamps:
                    x = y[...,-input_time_stamps:]
                    x_t = y_t[...,-input_time_stamps:]
                    x_tindicies = y_tindicies[...,-input_time_stamps:]

                elif output_time_stamps == input_time_stamps:
                    x = y #torch.cat((x[..., input_time_stamps:], out[...,:input_time_stamps]), dim=-1)
                    x_t = y_t #torch.cat((x_t[..., input_time_stamps:], y_t[...,:input_time_stamps]), dim=-1)
                    x_tindicies = y_tindicies #torch.cat((x_tindicies[..., input_time_stamps:], y_tindicies[...,:input_time_stamps]), dim=-1)
                
                elif output_time_stamps < input_time_stamps:
                    x = torch.cat((x[..., -(input_time_stamps-output_time_stamps):], y), dim=-1)
                    x_t = torch.cat((x_t[..., -(input_time_stamps-output_time_stamps):], y_t), dim=-1)
                    x_tindicies = torch.cat((x_tindicies[..., -(input_time_stamps-output_time_stamps):], y_tindicies), dim=-1)
                
                # x = torch.cat((x[..., next_input_time_stamps:], out[...,:next_input_time_stamps]), dim=-1)
                # x_t = torch.cat((x_t[..., next_input_time_stamps:], y_t[..., :next_input_time_stamps]), dim=-1)
                # x_tindicies = torch.cat((x_tindicies[..., next_input_time_stamps:], y_tindicies[...,:next_input_time_stamps]), dim=-1)

                    
            if b == 0:
                train_prediction = pred.clone()
            else:
                train_prediction = torch.cat((train_prediction,pred), dim=0)

    #test_l2_full = torch.mean((train_prediction-train_actual)**2, dim=[0,1] ).sum()
    #print("train_prediction, train_actual -->", train_prediction.shape, train_actual.shape)
    test_l2_full = torch.mean((train_prediction[...,:train_actual.shape[-1]] - train_actual)**2 )

    result = [test_l2_full, train_prediction[...,:train_actual.shape[-1]], train_actual, xy_tindicies]
    return result



# def roll_out_test_during_training( args, model, loader, timestamps, data_batch_type, no_of_samp, dt, horizon):
#     model.eval()
#     t_pass_test = args.next_input_time_stamps
#     input_range = args.input_time_stamps
#     total_range = args.t_resolution
#     no_of_input = args.input_time_stamps
#     t_pred_steps = args.output_time_stamps
#     timestamps = torch.tensor(timestamps).to(device)

#     with torch.no_grad():
#         #import pdb; pdb.set_trace()

#         for b, (data, u_super, x, parameters) in enumerate(loader):
#             #import pdb; pdb.set_trace()
#             #count += 1
#             data = data.to(device)
#             parameters = parameters[...,:3].to(device)
#             #import pdb; pdb.set_trace()
#             #data_batch_type = data_batch_type
#             #data_batch = data_random_batch(input_sample_type=data_batch_type, total_range = total_range,  no_of_samp=(args.batch_size_train, no_of_samp), dt_input =dt, t_pred_steps= t_pred_steps, horizon = horizon)
            
            
#             if data_batch_type == 1:
#                 #indicies = torch.sort(torch.randint(total_range, (data.shape[0], no_of_samp)))[0]
#                 indicies = torch.cat(( torch.sort(torch.randint(t_pred_steps,(1,t_pred_steps)))[0], torch.arange(t_pred_steps,total_range)[torch.sort(torch.randint(total_range-t_pred_steps,(1,no_of_samp-t_pred_steps)))[0]]), dim=-1)
#             elif data_batch_type == 2:
#                 rr = torch.ones((data.shape[0],total_range))
#                 indicies = (torch.arange(total_range)*rr)[:, ::dt][:, :no_of_samp].long()
#             elif data_batch_type == 3:
#                 #rr = torch.cat((torch.arange(100),torch.arange(102,250,3)), dim=-1).repeat((data.shape[0], 1)).long()
#                 #indicies = torch.cat((torch.arange(100),torch.arange(102,250,3)), dim=-1).repeat((data.shape[0], 1)).long()
#                 #indicies = torch.cat((torch.arange(0,150, 3),torch.arange(150,250,1)), dim=-1).repeat((data.shape[0], 1)).long()
#                 #indicies = torch.cat((torch.arange(0,100, 1),torch.arange(100,250,3)), dim=-1).repeat((data.shape[0], 1)).long()
#         #         rr = torch.tensor([[  0,   7,   7,  12,  30,  35,  36,  41,  43,  43,  44,  48,  51,  57,
#         #   58,  59,  64,  65,  66,  67,  73,  74,  75,  84,  87,  89,  92,  94,
#         #   96, 105, 105, 107, 108, 111, 117, 118, 119, 129, 142, 144, 146, 153,
#         #  154, 159, 168, 169, 170, 171, 171, 180, 184, 184, 187, 192, 193, 194,
#         #  198, 200, 202, 203, 205, 206, 206, 207, 210, 213, 219, 225, 232, 233,
#         #  240, 242, 245, 246, 249]])
#                 #rr = torch.cat((torch.arange(0,25, 1),torch.arange(27,250,3)), dim=-1)
#                 #rr = torch.cat(( torch.sort(torch.randint(25,(1,25)))[0], torch.arange(25,250)[torch.sort(torch.randint(225,(1,75)))[0]]), dim=-1)
#         #         rr = torch.tensor([[  1,   3,   4,   7,   7,   7,   7,   8,  11,  13,  15,  15,  15,  16,
#         #   17,  19,  20,  20,  21,  21,  21,  21,  22,  24,  24,  27,  34,  34,
#         #   36,  36,  42,  43,  52,  61,  63,  65,  65,  66,  68,  70,  76,  81,
#         #   86,  87,  90,  91,  91,  94,  95,  96,  98,  98,  98, 101, 103, 108,
#         #  108, 111, 118, 123, 126, 126, 128, 129, 130, 132, 141, 142, 148, 152,
#         #  157, 159, 162, 164, 168, 168, 170, 170, 173, 183, 188, 191, 200, 201,
#         #  201, 202, 203, 209, 210, 213, 214, 221, 225, 231, 231, 238, 239, 245,
#         #  246, 249]])
                
#         #         rr = torch.tensor([[  6,   7,   8,   8,   8,   9,   9,  10,  12,  13,  14,  15,  15,  15,
#         #   16,  16,  17,  21,  21,  21,  21,  23,  24,  24,  24,  25,  27,  27,
#         #   28,  29,  29,  29,  31,  31,  32,  33,  34,  35,  36,  37,  38,  39,
#         #   39,  42,  42,  44,  49,  50,  50,  52,  54,  54,  55,  59,  61,  61,
#         #   61,  62,  62,  63,  64,  64,  64,  65,  65,  66,  67,  67,  68,  71,
#         #   72,  75,  75,  76,  77,  78,  83,  87,  87,  89,  91,  91,  93,  98,
#         #   99,  99, 100, 101, 106, 106, 108, 112, 113, 114, 119, 122, 126, 129,
#         #  130, 131, 132, 133, 135, 140, 146, 147, 148, 149, 149, 149, 151, 152,
#         #  154, 155, 155, 157, 158, 158, 159, 159, 159, 160, 163, 164, 165, 165,
#         #  166, 168, 168, 170, 172, 173, 177, 177, 179, 179, 182, 182, 182, 185,
#         #  185, 185, 186, 188, 190, 192, 195, 196, 197, 199, 201, 202, 202, 203,
#         #  214, 216, 217, 217, 221, 221, 224, 225, 226, 230, 231, 236, 241, 241,
#         #  243, 243, 243, 243, 243, 243, 249]])

#         #         rr = torch.tensor([[  4,   4,   4,   5,   7,   7,   8,   8,   9,  10,  10,  11,  12,  14,
#         #   16,  16,  18,  18,  20,  20,  21,  21,  21,  24,  24,  26,  28,  29,
#         #   31,  35,  35,  36,  36,  37,  42,  43,  51,  54,  59,  59,  60,  61,
#         #   61,  66,  66,  66,  67,  68,  70,  71,  72,  73,  75,  78,  81,  87,
#         #   89,  90,  94,  96, 100, 104, 104, 105, 108, 109, 110, 124, 125, 133,
#         #  133, 133, 138, 139, 142, 142, 145, 147, 148, 149, 150, 152, 156, 156,
#         #  159, 161, 172, 172, 173, 173, 173, 173, 174, 174, 176, 177, 182, 183,
#         #  188, 189, 191, 195, 195, 199, 201, 203, 206, 209, 214, 215, 218, 218,
#         #  222, 223, 223, 226, 229, 232, 234, 236, 242, 243, 245, 247, 249]])
                
#         #         rr = torch.tensor([[  0,   0,   0,   4,   4,   4,   4,   5,   5,   6,   6,   7,   9,  10,
#         #   10,  10,  11,  12,  14,  15,  16,  16,  18,  21,  23,  27,  29,  30,
#         #   31,  32,  35,  36,  38,  38,  41,  41,  41,  44,  48,  51,  51,  51,
#         #   53,  54,  54,  54,  60,  62,  62,  63,  64,  66,  69,  70,  70,  75,
#         #   88,  89,  91, 101, 105, 106, 107, 108, 112, 119, 120, 122, 123, 132,
#         #  134, 139, 141, 145, 145, 149, 149, 149, 150, 151, 152, 155, 156, 156,
#         #  160, 160, 160, 164, 169, 170, 171, 171, 173, 178, 180, 183, 186, 188,
#         #  196, 197, 199, 201, 201, 201, 203, 207, 212, 213, 214, 214, 216, 218,
#         #  219, 220, 221, 228, 232, 234, 238, 238, 238, 239, 244, 244, 249]])

#         #         rr = torch.tensor([[  1,   3,   4,   5,   7,   7,   8,   9,  10,  11,  11,  11,  13,  13,
#         #   14,  16,  17,  17,  17,  18,  20,  22,  23,  24,  24,  27,  45,  48,
#         #   54,  55,  58,  65,  67,  69,  73,  77,  79,  84,  86,  87,  97, 103,
#         #  104, 105, 106, 106, 110, 111, 112, 113, 114, 117, 119, 143, 156, 160,
#         #  162, 168, 171, 174, 180, 186, 186, 186, 200, 201, 220, 222, 228, 230,
#         #  238, 241, 242, 245, 249]])


#                 rr = torch.cat((torch.arange(0,150,1),torch.arange(150,250,2)), dim=-1)
#                 #rr = torch.cat((torch.arange(0,50,1),torch.arange(50,250,4)), dim=-1)

#                 indicies = rr.repeat((data.shape[0], 1)).long()

#             elif data_batch_type == 4:
#         #         rr = torch.tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  14,
#         #   15,  16,  18,  18,  19,  19,  20,  22,  23,  27,  27,  30,  30,  30,
#         #   30,  31,  32,  32,  33,  33,  35,  36,  36,  39,  41,  42,  47,  48,
#         #   48,  50,  50,  54,  54,  57,  57,  57,  57,  60,  64,  67,  67,  67,
#         #   68,  69,  70,  71,  76,  76,  77,  78,  79,  81,  82,  83,  84,  86,
#         #   88,  91,  91,  92,  92,  93,  95,  95, 100, 100, 101, 102, 102, 102,
#         #  103, 103, 106, 106, 107, 108, 109, 110, 111, 113, 116, 117, 118, 118,
#         #  121, 121, 123, 125, 125, 125, 135, 136, 136, 137, 137, 137, 137, 137,
#         #  138, 139, 140, 140, 140, 140, 143, 144, 145, 147, 148, 149, 150, 150,
#         #  152, 152, 155, 156, 156, 157, 159, 160, 161, 166, 169, 170, 171, 171,
#         #  172, 173, 173, 174, 175, 176, 178, 180, 180, 182, 186, 187, 190, 191,
#         #  193, 194, 194, 194, 200, 200, 200, 201, 204, 207, 208, 210, 211, 211,
#         #  213, 213, 213, 218, 218, 218, 221, 222, 222, 222, 222, 224, 226, 227,
#         #  227, 227, 228, 229, 230, 232, 233, 234, 237, 242, 242, 243, 244, 246,
#         #  246, 246, 248, 249]])


#                 rr = torch.tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
#           14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  26,
#           27,  29,  31,  31,  33,  34,  35,  35,  36,  37,  38,  43,  43,  49,
#           51,  52,  53,  53,  53,  53,  55,  55,  56,  58,  61,  62,  62,  64,
#           66,  69,  70,  71,  72,  73,  74,  74,  78,  78,  79,  80,  83,  84,
#           84,  85,  85,  88,  88,  89,  91,  92,  92,  92,  95,  96,  98,  98,
#           98, 100, 100, 101, 101, 102, 103, 103, 103, 105, 108, 108, 109, 111,
#          112, 116, 117, 119, 119, 119, 122, 126, 129, 130, 131, 131, 132, 133,
#          134, 134, 134, 138, 139, 141, 146, 148, 149, 149, 150, 152, 153, 155,
#          155, 161, 162, 164, 167, 168, 168, 168, 169, 171, 174, 174, 176, 176,
#          176, 179, 180, 182, 182, 182, 186, 187, 187, 189, 189, 190, 190, 191,
#          194, 195, 195, 196, 196, 198, 203, 203, 209, 209, 210, 210, 210, 210,
#          211, 211, 213, 213, 214, 215, 216, 217, 219, 226, 228, 228, 228, 228,
#          229, 233, 234, 235, 236, 238, 239, 239, 240, 241, 242, 243, 243, 244,
#          245, 247, 249, 249]])

#         #         rr = torch.tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
#         #   14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  26,  28,  28,
#         #   32,  35,  37,  40,  43,  49,  51,  58,  59,  60,  63,  68,  68,  68,
#         #   69,  71,  71,  72,  74,  75,  81,  89,  98, 118, 121, 121, 125, 125,
#         #  128, 129, 130, 134, 135, 136, 136, 137, 138, 140, 142, 145, 148, 150,
#         #  151, 154, 158, 159, 166, 167, 168, 173, 182, 184, 190, 190, 190, 195,
#         #  208, 214, 220, 225, 225, 227, 227, 230, 230, 239, 245, 248, 249, 249,
#         #  249, 249]])
#         #         rr = torch.tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  12,  22,  25,
#         #   26,  27,  27,  38,  39,  41,  44,  47,  50,  52,  55,  56,  57,  66,
#         #   68,  71,  73,  75,  76,  77,  81,  83,  84,  84,  85,  94,  96, 102,
#         #  102, 104, 110, 112, 117, 118, 120, 122, 122, 123, 134, 139, 142, 149,
#         #  154, 154, 155, 159, 162, 162, 166, 169, 171, 175, 176, 177, 178, 183,
#         #  186, 189, 191, 198, 198, 200, 201, 202, 202, 203, 206, 207, 208, 209,
#         #  219, 223, 225, 226, 230, 230, 234, 235, 237, 237, 238, 238, 238, 241,
#         #  247, 249]])

            

                
#                 indicies = rr.repeat((data.shape[0], 1)).long()

#             #import pdb; pdb.set_trace()

#             xy = torch.gather(data, -1, indicies.unsqueeze(1).repeat((1,data.shape[1],1)).to(device) )

#             time_indicies =  indicies

#             xy_t = torch.ones_like(xy)[:,0,:].to(device)
#             xy_t = xy_t*timestamps[time_indicies]
#             xy_t = xy_t.unsqueeze(1).repeat(1,data.shape[1],1)

#             xy_tindicies = time_indicies.long()


#             time_stamps = [i for i in range(0, time_indicies.shape[-1]+1, t_pass_test)]


#             if b == 0:
#                 train_actual = xy[..., t_pred_steps:].clone()
#             else:
#                 train_actual = torch.cat((train_actual, xy[..., t_pred_steps:]), dim=0)

#             # input:
#             x = xy[..., : t_pred_steps]
#             x_t = xy_t[..., :t_pred_steps]
#             x_tindicies = xy_tindicies[..., :t_pred_steps ]
            

#             #import pdb; pdb.set_trace()
#             for t in range(len(time_stamps) - 2 ):
#                 y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
#                 y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
#                 y_tindicies = xy_tindicies[..., time_stamps[t+1]:time_stamps[t+2]]

#                     # print("\n")
#                     # print("x -->", time_stamps[t], time_stamps[t+1] )
#                     # print("x_t -->", x_t[0,0,:])
#                     # print("y -->", time_stamps[t+1], time_stamps[t+2] )
#                     # print("y_t -->", y_t[0,0,:])


#                     # if args.model_mode.startswith("constant_dt"):
#                     #     if args.dataset_name.endswith("E1"):
#                     #         out = model(x).to(device)
#                     #     elif args.dataset_name.endswith("E2"):
#                     #         out = model(torch.cat((x, parameters), dim=-1)).to(device)

#                     # if args.model_mode.startswith("variable_dt"):
#                     #     #model_operation = "add"
#                     #     if args.model_input_operation.startswith("add"):
#                     #         out = model( torch.cat((x, torch.zeros(x.shape).to(device)), dim = -1) + x_t ) # ADD
#                     #     elif  args.model_input_operation.startswith("concat"):
#                     #         out = model( torch.cat((x, x_t), dim=-1 ) ).to(device)  #CONCAT



#                 if args.model_mode.startswith("constant_dt"):
#                     if args.dataset_name.endswith("E1"):
#                         out = model(x).to(device)
#                     elif args.dataset_name.endswith("E2"):
#                         out = model(torch.cat((x, parameters), dim=-1)).to(device)


#                 if args.model_mode.startswith("variable_dt"):

#                     if args.model_input_operation.startswith("add"):
#                         x_x_t = x + x_t
#                         if args.dataset_name.endswith("E1"):
#                             out = model(torch.cat( (x_x_t,y_t), dim=-1)).to(device)
#                         elif args.dataset_name.endswith("E2"):
#                             out = model(torch.cat((x_x_t, y_t, parameters), dim=-1)).to(device)

#                     elif args.model_input_operation.startswith("concat"):
#                         if args.dataset_name.endswith("E1"):
#                             out = model( x, x_t, y_t ).to(device)
#                         elif args.dataset_name.endswith("E2"):
#                             out = model(torch.cat(( torch.cat((x, x_t, y_t), dim=-1 ), parameters), dim=-1)).to(device)

#                     elif args.model_input_operation.startswith("t_embed_attend"):
#                         if args.dataset_name.endswith("E1"):
#                             out = model( x.to(device), x_tindicies.to(device), y_tindicies.to(device) ).to(device)
#                         elif args.dataset_name.endswith("E2"):
#                             out = model(torch.cat(( torch.cat((x, x_t, y_t, parameters), dim=-1 ), parameters), dim=-1)).to(device)


#                 if t == 0:
#                     pred = out[...,:t_pass_test]
#                 else:
#                     pred = torch.cat((pred, out[...,:t_pass_test]), -1)


#                 x = torch.cat((x[..., t_pass_test:], out[...,:t_pass_test]), dim=-1)
#                 x_t = torch.cat((x_t[..., t_pass_test:], y_t[..., :t_pass_test]), dim=-1)
#                 x_tindicies = torch.cat((x_tindicies[..., t_pass_test:], y_tindicies[...,:t_pass_test]), dim=-1)

                    
#             if b == 0:
#                 train_prediction = pred.clone()
#             else:
#                 train_prediction = torch.cat((train_prediction,pred), dim=0)

#     test_l2_full = torch.mean((train_prediction-train_actual)**2, dim=[0,1] ).sum()

#     #import pdb; pdb.set_trace()
#     return test_l2_full, train_prediction, train_actual #, indicies






# def variable_roll_out_test_during_training( args, model, loader, timestamps, start_index):
#     model.eval()
#     t_pass_test = args.next_input_time_stamps
#     input_range = args.input_time_stamps
#     total_range = args.total_t_range
#     no_of_input = args.input_time_stamps
#     t_pred_steps = args.output_time_stamps
#     timestamps = torch.tensor(timestamps).to(device)
#     with torch.no_grad():

#         for b, (data, u_super, x, parameters) in enumerate(loader):   #change to tesst loader

#             test_data = data.to(device)
#             parameters = parameters[...,:3].to(device)

#             rand_input_type_test = 2
#             input_batch = Input_Batch(data=test_data, input_sample_type=rand_input_type_test, input_range=input_range, total_range = total_range,  no_of_input=no_of_input, dt_input=1)

#             rand_output_type_test = 2

#             random_array_y = [
#                 Output_Batch(input_batch.input_indicies, data=test_data, output_sample_type=rand_output_type_test, total_range=total_range, no_of_output=200), 
#             ]

#             #import pdb; pdb.set_trace()

#             for samp in range(len(random_array_y)):

#                 if rand_output_type_test == 2:
#                         no_of_output_test = random_array_y[samp].output_indicies.shape[0]


#                 x = input_batch.input
#                 y = random_array_y[samp].output
#                 time_indicies = torch.cat((input_batch.input_indicies, random_array_y[samp].output_indicies), dim=-1)

#                 xy = torch.cat((x,y), dim=-1)
#                 xy_t = torch.ones_like(xy).to(device)

#                 #import pdb; pdb.set_trace()
#                 xy_t = xy_t*timestamps[time_indicies]

#                 # if b == 0:
#                 #     train_actual = xy[..., t_pred_steps:].clone()
#                 # else:
#                 #     train_actual = torch.cat((train_actual, xy[..., t_pred_steps:]), dim=0)
            

#                 xy_t = torch.ones_like((xy)).to(device)
#                 xy_t = xy_t*timestamps[time_indicies]

#                 # x = xy[..., : t_pred_steps]
#                 # x_t = xy_t[..., :t_pred_steps]

#                 time_stamps = [i for i in range(0, len(time_indicies)+1, t_pass_test)]


#                 # data from is 25:
#                 time_stamps = time_stamps[start_index:]
#                 xy_t = xy_t[...,start_index*t_pred_steps:] 
#                 xy = xy[...,start_index*t_pred_steps:]


#                 if b == 0:
#                     train_actual = xy[..., t_pred_steps:].clone()
#                 else:
#                     train_actual = torch.cat((train_actual, xy[..., t_pred_steps:]), dim=0)

#                 # input:
#                 x = xy[..., : t_pred_steps]
#                 x_t = xy_t[..., :t_pred_steps]

#                 for t in range(len(time_stamps) - 2 ):
#                     y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
#                     y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]

#                     # print("\n")
#                     # print("x -->", time_stamps[t], time_stamps[t+1] )
#                     # print("x_t -->", x_t[0,0,:])
#                     # print("y -->", time_stamps[t+1], time_stamps[t+2] )
#                     # print("y_t -->", y_t[0,0,:])


#                     # if args.model_mode.startswith("constant_dt"):
#                     #     if args.dataset_name.endswith("E1"):
#                     #         out = model(x).to(device)
#                     #     elif args.dataset_name.endswith("E2"):
#                     #         out = model(torch.cat((x, parameters), dim=-1)).to(device)

#                     # if args.model_mode.startswith("variable_dt"):
#                     #     #model_operation = "add"
#                     #     if args.model_input_operation.startswith("add"):
#                     #         out = model( torch.cat((x, torch.zeros(x.shape).to(device)), dim = -1) + x_t ) # ADD
#                     #     elif  args.model_input_operation.startswith("concat"):
#                     #         out = model( torch.cat((x, x_t), dim=-1 ) ).to(device)  #CONCAT




#                     if args.model_mode.startswith("constant_dt"):
#                         if args.dataset_name.endswith("E1"):
#                             out = model(x).to(device)
#                         elif args.dataset_name.endswith("E2"):
#                             out = model(torch.cat((x, parameters), dim=-1)).to(device)


#                     if args.model_mode.startswith("variable_dt"):

#                         if args.model_input_operation.startswith("add"):
#                             x_x_t = torch.cat((x, torch.zeros(x.shape).to(device)), dim = -1) + x_t
#                             if args.dataset_name.endswith("E1"):
#                                 out = model(x_x_t).to(device)
#                             elif args.dataset_name.endswith("E2"):
#                                 out = model(torch.cat((x_x_t, parameters), dim=-1)).to(device)

#                         elif args.model_input_operation.startswith("concat"):
#                             if args.dataset_name.endswith("E1"):
#                                 out = model( torch.cat((x, x_t), dim=-1 )).to(device)
#                             elif args.dataset_name.endswith("E2"):
#                                 out = model(torch.cat(( torch.cat((x, x_t), dim=-1 ), parameters), dim=-1)).to(device)



#                     if t == 0:
#                         pred = out[...,:t_pass_test]
#                     else:
#                         pred = torch.cat((pred, out[...,:t_pass_test]), -1)

#                     if (t_pass_test < t_pred_steps) and (t == no_of_output_test - t_pred_steps + 1 - 1):
#                         pred = torch.cat((pred, out[...,t_pass_test: ]), -1)

#                     if t < no_of_output_test-t_pred_steps:
#                         x = torch.cat((x[..., t_pass_test:], out[...,:t_pass_test]), dim=-1)
#                         x_t = torch.cat((x_t[..., t_pass_test:], y_t[..., :t_pass_test]), dim=-1)

                        
#                 if b == 0:
#                     train_prediction = pred.clone()
#                 else:
#                     train_prediction = torch.cat((train_prediction,pred), dim=0)

#     test_l2_full = torch.mean((train_prediction-train_actual)**2, dim=[0,1] ).sum()

#     return test_l2_full, train_prediction, train_actual




# def test_only( args, model, loader, timestamps, start_index):
#     model.eval()

#     t_pass_test = args.next_input_time_stamps
#     input_range = args.input_time_stamps
#     total_range = args.total_t_range
#     no_of_input = args.input_time_stamps
#     t_pred_steps = args.output_time_stamps
#     timestamps = torch.tensor(timestamps).to(device)
#     with torch.no_grad():

#         for b, (data, u_super, x, parameters) in enumerate(loader):   #change to tesst loader

#             test_data = data.to(device)
#             parameters = parameters[...,:3].to(device)

#             rand_input_type_test = 2
#             input_batch = Input_Batch(data=test_data, input_sample_type=rand_input_type_test, input_range=input_range, total_range = total_range,  no_of_input=no_of_input, dt_input=1)

#             rand_output_type_test = 2

#             random_array_y = [
#                 Output_Batch(input_batch.input_indicies, data=test_data, output_sample_type=rand_output_type_test, total_range=total_range, no_of_output=200), 
#             ]

#             #import pdb; pdb.set_trace()

#             for samp in range(len(random_array_y)):

#                 if rand_output_type_test == 2:
#                         no_of_output_test = random_array_y[samp].output_indicies.shape[0]


#                 x = input_batch.input
#                 y = random_array_y[samp].output
#                 time_indicies = torch.cat((input_batch.input_indicies, random_array_y[samp].output_indicies), dim=-1)

#                 xy = torch.cat((x,y), dim=-1)
#                 xy_t = torch.ones_like(xy).to(device)

#                 xy_t = xy_t*timestamps[time_indicies]

#                 #import pdb; pdb.set_trace()



#                 #no_of_output_test = random_array_y[samp].output_indicies.shape[0]

#                 #output_indices_test = random_array_y[samp]

#                 #time_indicies_test = torch.cat((input_indicies_test.long(), output_indices_test.long() ), dim=-1)
                
#                 #import pdb; pdb.set_trace()
#                 #xy = test_data[ ..., time_indicies ]

#                 xy_t = torch.ones_like((xy)).to(device)
#                 xy_t = xy_t*timestamps[time_indicies]



#                 time_stamps = [i for i in range(0, len(time_indicies)+1, t_pass_test)]

#                 # data from is 25:
#                 time_stamps = time_stamps[start_index:]
#                 xy_t = xy_t[...,start_index*t_pred_steps:] 
#                 xy = xy[...,start_index*t_pred_steps:]


#                 if b == 0:
#                     train_actual = xy[..., t_pred_steps:].clone()
#                 else:
#                     train_actual = torch.cat((train_actual, xy[..., t_pred_steps:]), dim=0)
            

#                 x = xy[..., : t_pred_steps]
#                 x_t = xy_t[..., :t_pred_steps]


#                 #import pdb; pdb.set_trace()
#                 for t in range(len(time_stamps) - 2 ):
                    
#                     y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
#                     y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]

#                     # print("\n")
#                     # print("x -->", time_stamps[t], time_stamps[t+1] )
#                     # print("x_t -->", x_t[0,0,:])
#                     # print("y -->", time_stamps[t+1], time_stamps[t+2] )
#                     # print("y_t -->", y_t[0,0,:])
#                     #import pdb; pdb.set_trace()

#                     if args.model_mode.startswith("constant_dt"):
#                         if args.dataset_name.endswith("E1"):
#                             out = model(x).to(device)
#                         elif args.dataset_name.endswith("E2"):
#                             out = model(torch.cat((x, parameters), dim=-1)).to(device)

#                     if args.model_mode.startswith("variable_dt"):
#                         #model_operation = "add"
#                         if args.model_input_operation.startswith("add"):
#                             out = model( torch.cat((x, torch.zeros(x.shape).to(device)), dim = -1) + x_t ) # ADD
#                         elif  args.model_input_operation.startswith("concat"):
#                             out = model( torch.cat((x, x_t), dim=-1 ) ).to(device)  #CONCAT


#                     if t == 0:
#                         pred = out[...,:t_pass_test]
#                     else:
#                         pred = torch.cat((pred, out[...,:t_pass_test]), -1)

#                     if (t_pass_test < t_pred_steps) and (t == no_of_output_test - t_pred_steps + 1 - 1):
#                         pred = torch.cat((pred, out[...,t_pass_test: ]), -1)

#                     if t < no_of_output_test-t_pred_steps:
#                         x = torch.cat((x[..., t_pass_test:], out[...,:t_pass_test]), dim=-1)
#                         x_t = torch.cat((x_t[..., t_pass_test:], y_t[..., :t_pass_test]), dim=-1)

                        
#                 if b == 0:
#                     train_prediction = pred.clone()
#                 else:
#                     train_prediction = torch.cat((train_prediction,pred), dim=0)

#     test_l2_full = torch.mean((train_prediction-train_actual)**2, dim=[0,1] ).sum()

#     return test_l2_full, train_prediction, train_actual


# def test_during_training(args, proto, times_eval):
#     file_saved = "protocol_" + str(proto) +".pt"
#     import pdb; pdb.set_trace()
#     #saved_result = os.path.join(args.current_result_save_path, _1.pt"
#     saved_result = torch.load(   os.path.join(args.current_result_save_path, file_saved  ),  map_location=device )   

#     epoch = saved_result["saved_epoch"][-1]["epoch"]
#     model = saved_result["saved_epoch"][-1]["model"]


#     if args.input_shape_type.startswith("concat"):
#         model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps }
#     else:
#         raise TypeError("Specify the input shape type")
    
#     model_test =  FNO1d(model_hyperparameters["modes"], model_hyperparameters["width"], model_hyperparameters["input_size"], model_hyperparameters["output_size"]).to(device)
#     model_test.load_state_dict(model)
#     t_pred_steps = args.output_time_stamps
#     t_pass_test = args.output_time_stamps
#     no_of_input = args.input_time_stamps
#     testing_mode = "A_R"

#     # id_50 = torch.arange(50,250,4)
#     # id_50[-1]  = 249

#     #id_128 = torch.cat( (torch.arange(50,170,4), torch.arange(170,250,1)))
#     #id_128[-1]  = 249


#     #id_100 = torch.arange(51,250,2)


#     #input_indicies_test = torch.arange(0,50,5).to(device)
#     #input_indicies_test = torch.sort(torch.randint(0,50,(no_of_input,)))[0].to(device)  #
#     input_indicies_test = torch.arange(10).to(device)
#     #[id_100.to(device), id_128.to(device), torch.arange(50,250,1).to(device)]  #torch.arange(50,250,4).to(device)
#     #random_array_y =  [torch.arange(50,250,4).to(device), id_100.to(device), torch.arange(50,250,1).to(device)]

#     #test_l2_full = torch.zeros(len(random_array_y)).to(device)
    
#     random_array_y =  times_eval.copy()


#     # prediction_var_rand_in = []
#     # actual_var_rand_in = []

#     # prediction_var = []
#     # actual_var = []

#     for b,test_data in enumerate(train_loader):   #change to tesst loader
#         test_data = test_data[0].to(device)

#         rand_input_type_test = 2
#         #input_indicies_test = Input_Batch(data=test_data, input_sample_type=rand_input_type_test, input_range=input_range, total_range = total_range,  no_of_input=no_of_input, dt_input=1).input_indicies

#         rand_output_type_test = 2
#         # random_array_y = [
#         #     Output_Batch(input_indicies_test, data=test_data, output_sample_type=rand_output_type_test, total_range=total_range, no_of_output=50).output_indicies,
#         #     Output_Batch(input_indicies_test, data=test_data, output_sample_type=rand_output_type_test, total_range=total_range, no_of_output=100).output_indicies,
#         #     Output_Batch(input_indicies_test, data=test_data, output_sample_type=rand_output_type_test, total_range=total_range, no_of_output=200).output_indicies

#         # ]
#         # random_array_y = [
#         #     Output_Batch(input_indicies_test, data=test_data, output_sample_type=rand_output_type_test, total_range=total_range, no_of_output=200).output_indicies
#         # ]


#         for samp in range(len(random_array_y)):

#             if rand_output_type_test == 2:
#                     no_of_output_test = random_array_y[samp].shape[0]

#             #test_loss = 0
#             no_of_output_test = random_array_y[samp].shape[0]

#             output_indices_test = random_array_y[samp]
            
            
    
#             x = test_data[...,input_indicies_test]
#             y = test_data[...,output_indices_test]


#             if b == 0:
#                 train_actual = y.clone()
#             else:
#                 train_actual = torch.cat((train_actual,y), dim=0)


#             x_t = torch.ones((x.shape[0],x.shape[1], no_of_input + t_pred_steps)).to(device)
#             y_t = torch.ones((y.shape[0],y.shape[1], no_of_output_test - t_pred_steps)).to(device)


#             time_x = timestamps[input_indicies_test]
#             time_y = timestamps[output_indices_test]

#             x_t = x_t*torch.cat((time_x,time_y[:t_pred_steps]), dim=-1).to(device)
#             y_t = y_t*time_y[t_pred_steps:].to(device)

#             #print([i for i in range(0, no_of_output_test-t_pred_steps + 1, t_pass_test)])
#             for t in range(0, no_of_output_test - t_pred_steps + 1, t_pass_test):
#                 # print("\n")
#                 # print("t_range -->", range(t,t + t_pred_steps))
#                 y_true = y[..., t:t + t_pred_steps]


#                 #model_mode = "constant_dt"

#                 out = model_test( x ).to(device)


#                 if t == 0:
#                     pred = out[...,:t_pass_test]
#                 else:
#                     pred = torch.cat((pred, out[...,:t_pass_test]), -1)

#                 if (t_pass_test < t_pred_steps) and (t == no_of_output_test - t_pred_steps + 1 - 1):
#                     pred = torch.cat((pred, out[...,t_pass_test: ]), -1)

#                 if t < no_of_output_test-t_pred_steps:
#                     if testing_mode == "T_F":
#                         x = torch.cat((x[..., t_pass_test:], y_true[...,:t_pass_test]), dim=-1)
#                     elif testing_mode == "A_R":
#                         x = torch.cat((x[..., t_pass_test:], out[...,:t_pass_test]), dim=-1)
#                     else:
#                         raise TypeError("Choose training_mode: 'T_F' or 'A_R' ")

#                     x_t = torch.cat((x_t[..., t_pass_test:], y_t[..., t : t+t_pass_test]), dim=-1)

#             #print("pred -->", pred.shape)
                    
#             if b == 0:
#                 train_prediction = pred.clone()
#             else:
#                 train_prediction = torch.cat((train_prediction,pred), dim=0)

#     results = {"prediction" : train_prediction, "true" : train_actual, "time_eval" : times_eval}

#     #import pdb; pdb.set_trace()
#     torch.save(results, os.path.join(args.current_result_save_path, f"test_result_epoch_{epoch}.pt") )


