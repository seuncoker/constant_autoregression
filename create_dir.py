import numpy as np
import torch
import sys, os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))


from constant_autoregression.util import LpLoss, Printer, get_time, count_params, set_seed, create_current_results_folder, load_auguments, save_config_file
from constant_autoregression.argparser import arg_parse 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p = Printer(n_digits=6)


args = arg_parse()

arg_name = args.argument_file
#dir_name = args.dir_name
p.print(f"arg_nameeee: {arg_name}")
#p.print(f"dir_nameeee: {dir_name}")
args = load_auguments(args, arg_name)



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
print("folder_path: ", folder_path)
print("dataset_name: ", dataset_name)
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
p.print("\n")

# result_name = "result"
# folder_path =  args.current_dir_path
# path_result = os.path.join(folder_path, result_name)
# args.result_save_path = path_result
# try:
#     os.mkdir(path_result)
#     p.print(f"Folder '{result_name}' created successfully!")
# except FileExistsError:
#     p.print(f"Folder '{result_name}' already exists!")



# dataset_name =  args.dataset_name
# folder_path =  args.result_save_path 
# path_result = os.path.join(folder_path, dataset_name)
# args.result_save_path = path_result
# try:
#     os.mkdir(path_result)
#     p.print(f"Folder '{dataset_name}' created successfully!")
# except FileExistsError:
#     p.print(f"Folder '{dataset_name}' already exists!")


# analysis_name = args.analysis_type
# folder_path =  args.result_save_path 
# path_analysis_name = os.path.join(folder_path, analysis_name)
# args.result_save_path = path_analysis_name
# try:
#     os.mkdir(path_analysis_name)
#     p.print(f"Folder '{analysis_name}' created successfully!")
# except FileExistsError:
#     p.print(f"Folder '{analysis_name}' already exists!")


# datedate = str(datetime.now().date() )
# folder_path =  args.result_save_path 
# path_date = os.path.join(folder_path, datedate)
# args.current_date_save_path = path_date
# try:
#     os.mkdir(path_date)
#     p.print(f"Folder '{datedate}' created successfully!")
# except FileExistsError:
#     p.print(f"Folder '{datedate}' already exists!")


# result_name = args.experiment
# result_path = os.path.join(path_date, result_name)
# args.current_result_save_path = result_path
# try:
#     os.mkdir(result_path)
#     p.print(f"Folder '{result_name}' created successfully!")
# except FileExistsError:
#     p.print(f"Folder '{result_name}' already exists!")





#import pdb; pdb.set_trace()

with open(arg_name+".txt", "w") as f:
  f.write(args.current_result_save_path)

#create_current_results_folder(args)


#import pdb; pdb.set_trace()
#print("finishing...")