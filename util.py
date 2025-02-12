from datetime import datetime
import torch
from termcolor import colored
import sys, os

import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import random
import operator
from functools import reduce
from functools import partial
from torch.nn import functional as F
import json
from torch.utils.data import Dataset
from einops import rearrange
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """Set up seed."""
    if seed == -1:
        seed = None
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)



def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
    from time import localtime, strftime, time
    if precision == "second":
        string = strftime("%Y-%m-%d %H:%M:%S", localtime())
    elif precision == "millisecond":
        string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    if is_bracket:
        string = "[{}] ".format(string)
    if return_numerical_time:
        return string, time()
    else:
        return string


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


class Printer(object):
    def __init__(self, is_datetime=True, store_length=100, n_digits=3):
        """
        Args:
            is_datetime: if True, will print the local date time, e.g. [2021-12-30 13:07:08], as prefix.
            store_length: number of past time to store, for computing average time.
        Returns:
            None
        """
        
        self.is_datetime = is_datetime
        self.store_length = store_length
        self.n_digits = n_digits
        self.limit_list = []

    def print(self, item, tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=-1, precision="second", is_silent=False):
        if is_silent:
            return
        string = ""
        if is_datetime is None:
            is_datetime = self.is_datetime
        if is_datetime:
            str_time, time_second = get_time(return_numerical_time=True, precision=precision)
            string += str_time
            self.limit_list.append(time_second)
            if len(self.limit_list) > self.store_length:
                self.limit_list.pop(0)

        string += "    " * tabs
        string += "{}".format(item)
        if avg_window != -1 and len(self.limit_list) >= 2:
            string += "   \t{0:.{3}f}s from last print, {1}-step avg: {2:.{3}f}s".format(
                self.limit_list[-1] - self.limit_list[-2], avg_window,
                (self.limit_list[-1] - self.limit_list[-min(avg_window+1,len(self.limit_list))]) / avg_window,
                self.n_digits,
            )

        if banner_size > 0:
            print("=" * banner_size)
        print(string, end=end)
        if banner_size > 0:
            print("=" * banner_size)
        try:
            sys.stdout.flush()
        except:
            pass

    def warning(self, item):
        print(colored(item, 'yellow'))
        try:
            sys.stdout.flush()
        except:
            pass

    def error(self, item):
        raise Exception("{}".format(item))
    

p = Printer(n_digits=6)


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):

        num_examples = x.size()[0]
        res = x.size()[1]
        tsteps = x.size()[2]

        # x = x.permute(2,1,0)
        # y = y.permute(2,1,0)

        # diff_norms = torch.norm(x.reshape(tsteps,-1) - y.reshape(tsteps,-1), self.p, 1)
        # y_norms = torch.norm(y.reshape(tsteps,-1), self.p, 1)

        diff_norms = torch.norm(x - y, self.p, 1)
        y_norms = torch.norm(y, self.p, 1)



        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# def save_checkpoint(epoch, time_elapsed, loss, model, optimizer ):
#     state_dict = {
#         "epoch": epoch,
#         "time_elapsed": time_elapsed,
#         "loss": loss,
#         "model": model.state_dict(),
#         "optimizer": optimizer.state_dict(),
#     }
#     torch.save(state_dict, f"checkpoint_{epoch}.pt")



def return_checkpoint(epoch, time_elapsed, loss, model, optimizer ):
    state_dict = {
        "epoch": epoch,
        "time_elapsed": time_elapsed,
        "loss": loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    return state_dict





def dynamic_weight_loss(ep, epochs, const_val, max_forward_pass):
    tt = torch.arange(max_forward_pass).to(device)  # const = -0.5
    a = torch.exp(torch.tensor((-const_val*(max_forward_pass/epochs)*ep)).clone().detach())
    return torch.exp(-a*tt)


# def dynamic_weight_loss_sq(ep, epochs, const_val, max_forward_pass, no_f_pass):
#     tt = torch.arange(no_f_pass).to(device)
#     tt = tt*(max_forward_pass//no_f_pass) # const = -0.5
#     a = torch.exp(torch.tensor(-const_val*(max_forward_pass/epochs)*ep)).clone().detach()
#     return torch.exp(-a*(tt**2))

def dynamic_weight_loss_sq(ep, epochs, const_val, max_forward_pass, no_f_pass):
    tt = torch.arange(max_forward_pass).to(device)
    #tt = tt*(max_forward_pass//no_f_pass) # const = -0.5
    a = torch.exp(torch.tensor(-const_val*(max_forward_pass/epochs)*ep)).clone().detach()
    return torch.exp(-a*(tt**2))






def create_current_results_folder(args):
    # Define the desired folder name and path

    result_name = "result"
    folder_path =  args.current_dir_path
    path_result = os.path.join(folder_path, result_name)
    args.result_save_path = path_result
    try:
        os.mkdir(path_result)
        p.print(f"Folder '{result_name}' created successfully!")
    except FileExistsError:
        p.print(f"Folder '{result_name}' already exists!")



    dataset_name =  args.dataset_name
    folder_path =  args.result_save_path
    path_dataset_name = os.path.join(folder_path, dataset_name)
    args.result_save_path = path_dataset_name
    try:
        os.mkdir(path_dataset_name)
        p.print(f"Folder '{dataset_name}' created successfully!")
    except FileExistsError:
        p.print(f"Folder '{dataset_name}' already exists!")



    analysis_name = args.analysis_type
    folder_path =  args.result_save_path 
    path_analysis_name = os.path.join(folder_path, analysis_name)
    args.result_save_path = path_analysis_name
    try:
        os.mkdir(path_analysis_name)
        p.print(f"Folder '{analysis_name}' created successfully!")
    except FileExistsError:
        p.print(f"Folder '{analysis_name}' already exists!")



    subanalysis_name = args.subanalysis_type
    folder_path =  args.result_save_path 
    path_subanalysis_name = os.path.join(folder_path, subanalysis_name)
    args.result_save_path = path_subanalysis_name
    try:
        os.mkdir(path_subanalysis_name)
        p.print(f"Folder '{subanalysis_name}' created successfully!")
    except FileExistsError:
        p.print(f"Folder '{subanalysis_name}' already exists!")

    

    experiment_name = args.experiment
    folder_path =  args.result_save_path 
    path_experiment_name = os.path.join(folder_path, experiment_name)
    args.current_result_save_path = path_experiment_name
    try:
        os.mkdir(path_experiment_name)
        p.print(f"Folder '{experiment_name}' created successfully!")
    except FileExistsError:
        p.print(f"Folder '{experiment_name}' already exists!")



def save_config_file(args):
    filename_args = "config"
    with open(os.path.join(args.current_result_save_path, filename_args), 'w') as f:
        json.dump(vars(args), f, indent=4)



# Read the JSON file and extract arguments
def load_auguments(args, filename):
    #import pdb; pdb.set_trace()
    p.print(f"filename: {filename}")
    if filename == "arguments":
        filename_json = filename+".json"
    elif filename.endswith("config"):
        filename_json = filename
    elif filename.endswith("arguments_test"):
        filename_json = filename+".json"
    
    #filename_json = filename + ".json"
    #import pdb; pdb.set_trace()
    try:
        with open(filename_json, "r") as f:
            data = json.load(f)
            for key, value in data.items():
                if hasattr(args, key):
                    setattr(args, key, value)  # Set argument if it exists
    except FileNotFoundError:
        p.print("Warning: 'arguments.json' not found. Using default arguments.")
    
    #import pdb; pdb.set_trace()
    return args



def save_checkpoint(epoch, time_elapsed, loss, model, optimizer, prediction, actual ):
    state_dict = {
        "epoch": epoch,
        "time_elapsed": time_elapsed,
        "loss": loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "prediction": prediction, 
        "actual": actual,
    }
    torch.save(state_dict, f"checkpoint_{epoch}.pt")


def return_checkpoint(epoch, time_elapsed, learning_rate, error_info, model, optimizer ):
    state_dict = {
        "epoch": epoch,
        "time_elapsed": time_elapsed,
        "learning_rate": learning_rate,
        "error_info": error_info,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    return state_dict


def initialize_weights_xavier_uniform(model):
  for m in model.modules():
    if isinstance(m, nn.Linear):
      torch.nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)



def create_data(xy, xy_t, random_steps, t_pred_steps, horizon):

    x = torch.Tensor().to(device)
    x_t = torch.Tensor().to(device)
    y = torch.Tensor().to(device)
    y_t = torch.Tensor().to(device)

    for (ii, jj, kk) in zip(xy, xy_t, random_steps):
      xx = ii[..., kk - t_pred_steps : kk]
      xx_t = jj[..., kk - t_pred_steps : kk]

      yy = ii[..., kk : kk + (horizon*t_pred_steps)]
      yy_t = jj[..., kk : kk + (horizon*t_pred_steps)]

      x = torch.cat((x, xx[None,:]), 0).to(device)
      x_t = torch.cat((x_t, xx_t[None,:]), 0).to(device)

      y = torch.cat((y, yy[None,:]), 0).to(device)
      y_t = torch.cat((y_t, yy_t[None,:]), 0).to(device)

    return torch.cat((x,y),dim=-1), torch.cat((x_t, y_t),dim=-1)






class batch_time_sampling():
    """Object for holding a batch of data with mask during training."""

    def __init__(self, choice, total_range, no_of_samp, t_pred_steps, dt=1):
        if choice == 1:
            self.indicies = self.input_indicies_1(total_range, no_of_samp, dt)
        elif choice == 2:
            self.indicies = self.input_indicies_2(total_range, no_of_samp)
        elif choice == 3:
            self.indicies = self.input_indicies_3(total_range, no_of_samp)
        else:
            raise TypeError("Specify input_sample_type: 1 (non_independent sampes ) OR 2 (Independent samples )")


    @staticmethod
    def input_indicies_1(total_range, no_of_samp, dt):
        """
        random constant timestep sampling
        """
        #init_time_stamp_range = torch.tensor([t for t in range(0, total_range -  no_of_samp[1]*dt)])
        init_time_stamp_range = torch.tensor([t for t in range(0, (total_range -  (no_of_samp[1] + ((no_of_samp[1] -1)*(dt-1)) ) + 1 ))])
        random_steps = init_time_stamp_range[torch.randint(len(init_time_stamp_range), (no_of_samp[0],))]

        indicies = torch.ones((no_of_samp) )
        for i in range(no_of_samp[0]):
            start = random_steps[i]
            indicies[i] = torch.arange(start,total_range,dt)[:no_of_samp[1]]
        


        """
        sequential constant timestep sampling
        """

        # indicies = torch.arange(0, total_range, dt)[:no_of_samp[1]].unsqueeze(0)
        # indicies = indicies.repeat(no_of_samp[0],1)

        return indicies.long()
    


    @staticmethod
    def input_indicies_2(total_range, no_of_samp):
        """
        variable time steps
        """
        init_time_stamp_range = torch.tensor([t for t in range(0, total_range -  no_of_samp[1])])
        random_steps = init_time_stamp_range[torch.randint(len(init_time_stamp_range), (no_of_samp[0],))]

        indicies = torch.ones((no_of_samp) )
        for i in range(no_of_samp[0]):
            start = random_steps[i]
            final_time_stamp_range = torch.tensor([t for t in range(start +  no_of_samp[1] , total_range)])
            end = final_time_stamp_range[torch.randperm(len(final_time_stamp_range))[0]]
            indicies[i] = torch.sort(start + torch.randint(end-start, (1,no_of_samp[1]) ))[0]
        return indicies.long()



    @staticmethod
    def input_indicies_3(total_range, no_of_samp):
        """
        generate n random input samples from the range (250 )
        for testing......
        """
        indicies = torch.sort(torch.randint(total_range-1, (no_of_samp[0], no_of_samp[1] )), dim=-1)[0]
        last_indcies = torch.full((no_of_samp[0], 1), total_range-1, dtype=torch.long)
        indicies = torch.cat((indicies, last_indcies), dim=-1)
        #indicies = torch.cat(( torch.sort(torch.randint(t_pred_steps,(no_of_samp[0],t_pred_steps)))[0], torch.arange(t_pred_steps,total_range)[torch.sort(torch.randint(total_range-t_pred_steps,(no_of_samp[0],no_of_samp[1]-t_pred_steps)))[0]]), dim=-1)
        return indicies.long()
    

    @staticmethod
    def input_indicies_4(total_range, no_of_samp, t_pred_steps):
        """
        generate n random input samples from the range (250 )
        for testing......
        """
        indicies = torch.cat(( torch.sort(torch.randint(t_pred_steps,(no_of_samp[0],t_pred_steps)))[0], torch.arange(t_pred_steps,total_range)[torch.sort(torch.randint(total_range-t_pred_steps,(no_of_samp[0],no_of_samp[1]-t_pred_steps)))[0]]), dim=-1)
        return indicies.long()


    def input_indicies_5(total_range, no_of_samp, dt):
        """
        sequential constant timestep sampling
        """
        #init_time_stamp_range = torch.tensor([t for t in range(0, total_range -  no_of_samp[1]*dt)])
        # init_time_stamp_range = torch.tensor([t for t in range(0, (total_range -  (no_of_samp[1] + ((no_of_samp[1] -1)*(dt-1)) ) + 1 ))])
        # random_steps = init_time_stamp_range[torch.randint(len(init_time_stamp_range), (no_of_samp[0],))]
        
        indicies = torch.arange(0, total_range, dt)[:no_of_samp[1]].unsqueeze(0)
        indicies = indicies.repeat(no_of_samp[0],1)

        # indicies = torch.ones((no_of_samp) )
        # for i in range(no_of_samp[0]):
        #     start = random_steps[i]
        #     indicies[i] = torch.arange(start,total_range,dt)[:no_of_samp[1]]
        return indicies.long()



def subsequent_mask(size):
    "Mask out subsequent positions."
    #size =tgt.shape[1]
    #size = 100
    #print("size -->",size)
    #attn_shape = (1,1, size, size)
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )#.to(device)
#print("subsequent_mask -->", subsequent_mask.shape)
    return subsequent_mask == 0





class test_batch_time_sampling():
    """Object for holding a batch of data with mask during training."""

    def __init__(self, choice, total_range, no_of_samp, t_pred_steps, dt=1):
        if choice == 1:
            self.indicies = self.input_indicies_1(total_range, no_of_samp, dt)
        elif choice == 2:
            self.indicies = self.input_indicies_2(total_range, no_of_samp)
        elif choice == 3:
            self.indicies = self.input_indicies_3(total_range, no_of_samp)
        else:
            raise TypeError("Specify input_sample_type: 1 (non_independent sampes ) OR 2 (Independent samples )")


    @staticmethod
    def input_indicies_1(total_range, no_of_samp, dt):
        """
        constant timestep sampling
        """
        #init_time_stamp_range = torch.tensor([t for t in range(0, total_range -  no_of_samp[1]*dt)])
        #init_time_stamp_range = torch.tensor([t for t in range(0, (total_range -  (no_of_samp[1] + ((no_of_samp[1] -1)*(dt-1)) ) + 1 ))])
        #random_steps = init_time_stamp_range[torch.randint(len(init_time_stamp_range), (no_of_samp[0],))]
        indicies = torch.arange(0,total_range,dt)[:no_of_samp[1]]
        indicies = indicies.repeat((no_of_samp[0], 1)).long()
        return indicies
    


    @staticmethod
    def input_indicies_2(total_range, no_of_samp):
        """
        generate n random input samples from the range (250 )
        for testing......
        """
        first_indcies = torch.full((no_of_samp[0], 1), 0, dtype=torch.long)
        indicies = torch.sort(torch.randint(total_range-2, (no_of_samp[0], no_of_samp[1]-2 )), dim=-1)[0] + 1
        last_indcies = torch.full((no_of_samp[0], 1), total_range-1, dtype=torch.long)
        indicies = torch.cat((first_indcies, indicies, last_indcies), dim=-1)
        #indicies = torch.cat(( torch.sort(torch.randint(t_pred_steps,(no_of_samp[0],t_pred_steps)))[0], torch.arange(t_pred_steps,total_range)[torch.sort(torch.randint(total_range-t_pred_steps,(no_of_samp[0],no_of_samp[1]-t_pred_steps)))[0]]), dim=-1)
        return indicies.long()
    


    @staticmethod
    def input_indicies_3(total_range, no_of_samp):
        """
        split input and output sampling
        """
        ############ B1
        #indicies = torch.cat(( torch.sort(torch.randint(t_pred_steps,(no_of_samp[0],t_pred_steps)))[0], torch.arange(t_pred_steps,total_range)[torch.sort(torch.randint(total_range-t_pred_steps,(no_of_samp[0],no_of_samp[1]-t_pred_steps)))[0]]), dim=-1)
        #indicies = torch.cat((torch.arange(0,66,1), torch.arange(67,201,4)), )
        #indicies = torch.cat((torch.arange(0,65,2), torch.arange(68,201,2)), )
        #indicies = torch.cat(( torch.arange(10), torch.arange(11,201,2)))
        #indicies = torch.cat((torch.arange(0,65,3), torch.arange(66,70,1), torch.arange(71,201,3)), )
        #indicies = torch.cat(( torch.arange(10), torch.arange(10,160,3), torch.arange(161,201,1)))
        indicies = torch.cat((torch.arange(10), torch.arange(10,50,1), torch.arange(53,201,3)))
        #indicies = torch.cat((torch.arange(10), torch.arange(10,50,1), torch.arange(54,201,5)))
        #indicies = torch.cat((torch.arange(10), torch.arange(10,50,1), torch.arange(65,201,15)))
        #indicies = torch.cat((torch.arange(0,170,4),          torch.arange(174,201,1)))


        ######## E1
        #indicies = torch.cat(( torch.arange(10), torch.arange(10,190,3), torch.arange(190,250,1)))
        #indicies = torch.cat(( torch.arange(10), torch.arange(10,69,1), torch.arange(69,250,3)))
        #indicies =  torch.cat(( torch.arange(10), torch.arange(11,250,2)))
        
        indicies = indicies.unsqueeze(0).repeat(no_of_samp[0],1)
        return indicies.long()





def create_next_data(x, x_t, out, y, y_t, t_pass_train):
    #print("x_concat_out -->",x[..., t_pass_train:].shape, out[...,:t_pass_train].shape)
    x = torch.cat((x[..., t_pass_train:], out[...,:t_pass_train]), dim=-1)
    x_t = torch.cat((x_t[..., t_pass_train:], y_t[...,:t_pass_train]), dim=-1)
    return x, x_t



def train_print_time(args, ep, last_epoch_no, s, time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, loss, f_pass_weights_random, a_l, rand_horizon, rand_horizon_grad, input_time_stamps):
    if ep == (last_epoch_no+1) and s < 10 :
    #if ep == (last_epoch_no+1):
    #if s == 0 :
        p.print(f"horizon --> {rand_horizon}" )
        p.print(f"horizon_grad --> {rand_horizon_grad}" )
        p.print(f"x --> {time_stamps[t],input_time_stamps+time_stamps[t]}" )
        p.print(f"x_t --> {x_tindicies[:3,:5]}")
        p.print(f"y --> {input_time_stamps+time_stamps[t], input_time_stamps+time_stamps[t+1]} ")
        p.print(f"y_t --> {y_tindicies[:3,:5]} ")
        p.print(f"f_pass_weights[{a_l}] ->, {f_pass_weights_random[:3,a_l]}")
        p.print(f"loss ->, {loss}")
        p.print("\n")


def test_print_time(args, b,time_stamps, t, x_t, y_t, x_tindicies, y_tindicies, input_time_stamps):
    if b == 0:
        p.print(f"x --> {time_stamps[t],input_time_stamps+time_stamps[t]}" )
        p.print(f"x_t --> {x_tindicies[:3,:5]}")
        p.print(f"y --> {input_time_stamps+time_stamps[t],input_time_stamps+time_stamps[t+1]} ")
        p.print(f"y_t --> {y_tindicies[:3,:5]} ")
        p.print("\n")






class FFNO_Normalizer(nn.Module):
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
    

def pearson_correlation(input: torch.Tensor, target: torch.Tensor, reduce_batch: bool = True):
    B = input.size(0)
    T = input.size(1)
    input = input.reshape(B, T, -1)
    target = target.reshape(B, T, -1)
    input_mean = torch.mean(input, dim=(2), keepdim=True)
    target_mean = torch.mean(target, dim=(2), keepdim=True)
    # Unbiased since we use unbiased estimates in covariance
    input_std = torch.std(input, dim=(2), unbiased=False)
    target_std = torch.std(target, dim=(2), unbiased=False)

    corr = torch.mean((input - input_mean) * (target - target_mean), dim=2) / (input_std * target_std).clamp(
        min=torch.finfo(torch.float32).tiny
    )  # shape (B, T)
    if reduce_batch:
        corr = torch.mean(corr, dim=0)
    return corr





def result_loss_mse(file_loc):
    results = []
    for i in range(len(file_loc)):
        data = torch.load(file_loc[i], map_location=device)
        result = {}
        if "cons_ro_250" in data.keys():
            actual_ro = data["cons_ro_250"][2]
            prediction_ro = data["cons_ro_250"][1][..., :actual_ro.shape[-1]]
            result["test_rollout"] = torch.mean((prediction_ro-actual_ro)**2)


        if "cons_oto_250" in data.keys():
            #prediction_ro = data["cons_oto_250"][1]
            actual_ro = data["cons_oto_250"][2]
            prediction_ro = data["cons_oto_250"][1][..., :actual_ro.shape[-1]]
            result["test_onestep"] = torch.mean((prediction_ro-actual_ro)**2)
        
        if "train_cons_ro_250" in data.keys():
            #prediction_ro = data["train_cons_ro_250"][1]
            actual_ro = data["train_cons_ro_250"][2]
            prediction_ro = data["train_cons_ro_250"][1][..., :actual_ro.shape[-1]]
            result["train_rollout"] = torch.mean((prediction_ro-actual_ro)**2)

        if "train_cons_oto_250" in data.keys():
            #prediction_ro = data["train_cons_oto_250"][1]
            actual_ro = data["train_cons_oto_250"][2]
            prediction_ro = data["train_cons_oto_250"][1][..., :actual_ro.shape[-1]]
            result["train_onstep"] = torch.mean((prediction_ro-actual_ro)**2)
        print(result)
        results.append(result)                        
    return results


def result_correlation_threshold(file_loc, threshold):
    results = []
    time = np.linspace(0,4,250)*(100/4)
    for i in range(len(file_loc)):
        data = torch.load(file_loc[i], map_location=device)
        result = {}
        if "cons_ro_250" in data.keys():
            #prediction_ro = data["cons_ro_250"][1]
            actual_ro = data["cons_ro_250"][2]
            prediction_ro = data["cons_ro_250"][1][..., :actual_ro.shape[-1]]
            corr = pearson_correlation(prediction_ro.permute(0,2,1), actual_ro.permute(0,2,1) )
            result["test_rollout"] = time[torch.where(corr < threshold)[0][0]]
        
        if "train_cons_ro_250" in data.keys():
            #prediction_ro = data["train_cons_ro_250"][1]
            actual_ro = data["train_cons_ro_250"][2]
            prediction_ro = data["train_cons_ro_250"][1][..., :actual_ro.shape[-1]]
            corr = pearson_correlation(prediction_ro.permute(0,2,1), actual_ro.permute(0,2,1) )
            result["train_rollout"] = time[torch.where(corr < threshold)[0][0]]


        results.append(result)                       
    return results


def k_transition(n, T, N, delta = 0.2):
    k_trans = (N/2)*(1 + math.tanh((n/N - 0.5)/delta) )
    return T*(k_trans/N)  # 250*(k_transition(n,N, delta = 0.2)/ N)




def normalized_rmse(predictions, true_values):
  """
  Calculates the Normalized Root Mean Squared Error (nRMSE).

  Args:
    predictions: A array containing the predicted values.
    true_values: A array containing the true values.

  Returns:
    The normalized root mean squared error.
  """
  #print(predictions.shape, true_values.shape)

#   rmse = torch.sqrt(torch.sum((predictions - true_values) ** 2  ) )
  
#   nrmse = rmse /   torch.sqrt(torch.sum((true_values) ** 2) )    #torch.std(true_values)

  rmse = torch.sqrt(torch.mean((predictions - true_values) ** 2, dim = 1 ) )
  nrmse = rmse / torch.sqrt(torch.mean((true_values) ** 2, dim = 1) )    
  
  return torch.mean(nrmse)