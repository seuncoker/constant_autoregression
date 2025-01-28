import argparse
import sys, os
from pathlib import Path
from typing import List
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))


def arg_parse():
    parser = argparse.ArgumentParser(description='constant_Autoregression argparse.')

    ### RANDOMNESS
    parser.add_argument('--seed', type=int,
                        help='number for seeed')
    

    #### TASK
    parser.add_argument('--mode', type=str,
                        help=' Train or Test mode')
    parser.add_argument('--analysis_type', type=str,
                        help=' type of analysis ')
    parser.add_argument('--subanalysis_type', type=str,
                        help=' type of sub analysis ')
    parser.add_argument('--experiment', type=str,
                        help=' Name of experiment ')
    


    ### TEST MODE SPECIFCIATION
    parser.add_argument('--test_only_path', type=str,
                        help=' Test only file path')
    parser.add_argument('--test_only_protocol_no', type=int,
                        help=' Test only file path')  
    parser.add_argument('--test_only_epoch_index', type=int,
                        help=' Test only file path')  
    


    ### DATASET
    parser.add_argument('--dataset_name', type=str,
                        help=' Type of dataset. Choose from: "E1, E2, E3"')
    parser.add_argument('--dataset_train_path', type=str,
                        help=' URL of train datasets"')
    parser.add_argument('--dataset_valid_path', type=str,
                        help=' URL of valid datasets"')
    parser.add_argument('--dataset_test_path', type=str,
                        help=' URL of test datasets"')
    
    parser.add_argument('--t_resolution', type=int,
                        help='time resolution')
    parser.add_argument('--t_resolution_train', type=int,
                        help='time resolution')
    parser.add_argument('--t_resolution_test', type=int,
                        help='time resolution')
    parser.add_argument('--t_resolution_valid', type=int,
                        help='time resolution')
    
    parser.add_argument('--x_resolution', type=int,
                        help='spatial resolution')
    parser.add_argument('--timestamps', type=int,
                        help='time resolution values')
    parser.add_argument('--timestamps_test', type=int,
                        help='time resolution values for test data')  
    parser.add_argument('--timestamps_valid', type=int,
                    help='time resolution values for valid data')  
    parser.add_argument('--no_parameters', type=int,
                        help='number of pde parameters except spatial and time') 
                   

    ### TRAIN_DATA
    parser.add_argument('--n_train', type=int,
                        help='The first n_train examples will be used for the dataset. If -1, will use the full dataset.')
    parser.add_argument('--n_test', type=int,
                    help='The first n_test examples will be used for the dataset. If -1, will use the full dataset.')
    parser.add_argument('--batch_size_train', type=int,
                        help=' Training batch size.')
    parser.add_argument('--batch_size_test', type=int,
                        help=' Training batch size.')
    parser.add_argument('--time_stamps', type=int,
                    help=' time stamps for training.')
    parser.add_argument('--time_stamps_test', type=int,
                    help=' time stamps for testing.')    


    ### FILE LOCATION
    parser.add_argument('--root_dir', type=Path,
                        help='The direcotry of the project')
    parser.add_argument('--current_dir_path', type=str,
                        help='location of current working directory') 
    

    ### MODEL SPECIFICATION
    parser.add_argument('--model_type', type=str,
                        help='type of model')
    parser.add_argument('--fno_hidden_dim', type=int,
                        help='hidden dimesion of layers')
    parser.add_argument('--fno_hidden_layers', type=int,
                        help='Number of layers')
    parser.add_argument('--fno_modes', type=int,
                        help='Number of fourier modes')
    parser.add_argument('--pretrained_model', type=str,
                        help='location of pretrained model')
    parser.add_argument('--model_initialise_type', type=str,
                        help='type of initialise: random, xavier_uniform ')
    

    ### TIME INFORMATION
    parser.add_argument('--time_prediction', type=str,
                        help='constant or variable model')
    parser.add_argument('--time_conditioning', type=str,
                        help='addition or concatenate')
    parser.add_argument('--predict_difference', type=bool,
                        help='predict time difference')


    ### VARIABLE_TIME_STEP_PREDICTION
    parser.add_argument('--n_tsamples', type=List,
                        help='number of samples from time')
    parser.add_argument('--time_sampling_type', type=List,
                        help='int specifying the type of sampling')    

    ### CONSTANT_TIME_STEP_PREDICTION
    parser.add_argument('--dt_step', type=List,
                        help='number of time_skip for constant dt model') 

    ### MULITSTEP
    parser.add_argument('--input_time_stamps', type=int,
                        help='number of input time stamps')
    parser.add_argument('--output_time_stamps', type=int,
                        help='number of output time stamps per forward pass')
    parser.add_argument('--next_input_time_stamps', type=int,
                        help='number of output time stamps to pass for the next prediction (max: output_time_stamps )')


    ### OPTIMISER
    parser.add_argument('--initialise_optimiser', type=list,
                        help='init_optimiser')
    parser.add_argument('--optimizer_type', type=list,
                        help='type of optimiser')
    parser.add_argument('--learning_rate', type=list,
                        help='optimiser learning rate')
    parser.add_argument('--weight_decay', type=list,
                        help='optimiser weight decay')
    parser.add_argument('--min_learning_rate', type=list,
                    help='minimum learning_rate')
    

    ### SHEDULER
    parser.add_argument('--sheduler_type', type=list,
                        help='type of sheduler')
    parser.add_argument('--sheduler_step', type= list,
                        help='sheduler_step') 
    parser.add_argument('--sheduler_gamma', type= list,
                        help='sheduler_factor')
    parser.add_argument('--sheduler_change', type= list,
                    help='when to change learning rate: iteration or epoch')
    parser.add_argument('--cos_anneal_T_max', type= list,
                    help='iteration from max to min')
    

    
    ### TRAINING PROTOCOLS
    parser.add_argument('--new_training', type=bool,
                        help='True - New Training, False - Continue training')
    parser.add_argument('--training_protocol_type', type=str,
                        help='type of training strategy: training_loop by batch or by sequence') 
    parser.add_argument('--number_of_training_protocol', type=int,
                        help='number of training protocols')    
    parser.add_argument('--epochs', type=list,
                        help='number of training epochs')
    parser.add_argument('--iter_per_epochs', type=list,
                        help='iteration per epoch')
    parser.add_argument('--training_loop', type=str,
                        help='types of training loop: random_time_sampling, ')        


    ### LOSS FUNCTION
    parser.add_argument('--loss_train_type', type=str,
                        help='type of training loss function')
    parser.add_argument('--loss_test_type', type=str,
                        help='type of test loss function')



    ## DYNAMIC LOSS WEIGHTING
    parser.add_argument('--dynamic_loss_weight_per_fpass', type=list,
                        help='loss weighting for each forward pass through the model: True or False for each training protocol')
    parser.add_argument('--dynamic_loss_weight_per_fpass_constant_parameter', type=list,
                    help='fpass constant parameter values')
    parser.add_argument('--dynamic_loss_weight_per_fpass_reversed', type=list,
                        help='loss weighting for each forward pass through the model: True or False for each training protocol')    
    parser.add_argument('--dynamic_loss_weight_per_tstamp', type=list,
                        help='loss weighting for each tstamps : True or False for each training protocol')  
    parser.add_argument('--dynamic_loss_weight_per_tstamp_constant_parameter', type=list,
                        help='tstamp_constant parameter values')  
    parser.add_argument('--dynamic_loss_weight_per_fpass_type', type=str,
                        help='dynamic weighting type: "local or global')



    ### SAVE
    parser.add_argument('--epoch_save_interval', type=str,
                        help='epoch_save_interval')
    parser.add_argument('--epoch_print_interval', type=str,
                        help='epoch_print_interval')
    parser.add_argument('--result_save_path', type=str,
                        help='location for saving results') 
    parser.add_argument('--current_result_save_path', type=str,
                        help='location for saving current results') 
    parser.add_argument('--current_date_save_path', type=str,
                        help='location for saving current date') 



    ### AUTOREGRESSION
    parser.add_argument('--horizon', type=list,
                        help='number of autoregressive rollout')
    parser.add_argument('--horizon_type', type=str,
                        help='type of horizon: variable or constant')
    parser.add_argument('--random_horizon', type=str,
                        help='randomly select the horizon by specifying a maximum horizon of horizon')    

    ## PUSH FORWARD
    parser.add_argument('--push_forward', type=list,
                        help='error injection by model prediction')
  
    parser.add_argument('--push_forward_parameter', type=list,
                        help=' number of error injection by the model at the start of autoregression') 
    
    parser.add_argument('--push_forward_parameter_random', type=bool,
                        help='randomly sample the number of model injection by specifying push_forward_parameter ')  


    parser.add_argument('--noise', type=list,
                        help='noise true and false')
  
    parser.add_argument('--noise_std', type=list,
                        help=' value of noise') 
    
    parser.add_argument('--norm', type=bool,
                        help='randomly sample the number of model injection by specifying push_forward_parameter ')  
    parser.add_argument('--normalise_parameters', type=bool,
                        help='normalise input parameters or not ')  

    

    parser.set_defaults(
        
        ### RANDOMNESS
        seed = 42,


        #### TASK
        mode = "train",
        analysis_type = "analysis",
        subanalysis_type = "sub_analysis",
        experiment = "test",


        ### TEST MODE SPECIFCIATION
        test_only_path = None,
        test_only_protocol_no = None,
        test_only_epoch_index = None,


        ### DATASET
        dataset_name = None,
        dataset_train_path = None,
        dataset_valid_path = None,
        dataset_test_path = None,
        t_resolution = None,
        t_resolution_train = None,
        t_resolution_test = None,
        t_resolution_valid = None,
        x_resolution = None,
        timestamps = None,
        timestamps_test = None,
        no_parameters = None,
        

        ### TRAIN_DATA
        n_train = 1,
        n_test = 1,
        batch_size_train = 1,
        batch_size_test = 1,


        ### FILE LOCATION
        root_dir = None,
        current_dir_path = os.getcwd(),


        ### MODEL SPECIFICATION
        model_type = None,
        fno_hidden_dim = None,
        fno_hidden_layers = None,
        fno_modes = None,
        pretrained_model = None,
        model_initialise_type = None,


        ### TIME INFORMATION
        time_prediction = None,
        time_conditioning = None,
        time_sampling_choice = None,
        predict_difference = None,



        ### VARIABLE_TIME_STEP_PREDICTION
        n_tsamples = [],

        ### CONSTANT_TIME_STEP_PREDICTION
        dt_step = 1,

        ### MULITSTEP
        input_time_stamps = None,
        output_time_stamps = None,
        next_input_time_stamps = None,


        ### OPTIMISER
        initialise_optimiser = [],
        optimiser_type = [],
        learning_rate = [],
        weight_decay = [],
        min_learning_rate = [],


        ### SHEDULER
        sheduler_type = [],
        sheduler_step = [],
        sheduler_gamma = [],
        sheduler_change = [],
        cos_anneal_T_max = [],




        ### TRAINING PROTOCOLS
        new_training =  None,
        training_protocol_type = None,
        number_of_training_protocol =  None,
        training_loop = None,
        epochs = [],
        iter_per_epochs = [],


        ### LOSS
        loss_train_type = "l2",
        loss_test_type = "l2",
        



        ## DYNAMIC LOSS WEIGHTING
        # dynamic_loss_weight_per_fpass = [True, True, False, False],
        # dynamic_loss_weight_per_fpass_constant_parameter = [0.50, 0.51, 0.52, 0.51 ],
        # dynamic_loss_weight_per_fpass_reversed = [False, False, False, False],
        # dynamic_loss_weight_per_tstamp = [False, False, False, False],
        # dynamic_loss_weight_per_tstamp_constant_parameter = [0.51, 0.50, 0.51, 0.50 ],
        # dynamic_loss_weight_per_fpass_type = None,


        
        ### SAVE
        result_save_path = "",
        current_date_save_path = "",
        current_result_save_path = "",
        epoch_save_interval = 1,
        epoch_print_interval = 1,


        ### AUTOREGRESSION
        horizon = [1,],
        horizon_type = "constant",
        random_horizon = False,


        ## PUSH FORWARD
        push_forward = None,
        push_forward_parameter_random = None,
        push_forward_parameter = None,
 

        ### Noise
        noise = False, #args.noise,
        noise_std = 0.01, # args.noise_std,
        norm = False, #args.norm,
        normalise_parameters = False,
        
        
        training_protocols = [
            {"epochs": None,
             "iter_per_epochs": None,

            "random_horizon": None,
            
            "push_forward": None,
            "push_forward_parameter_random": None,
            "push_forward_parameter": 1,

             "dynamic_loss_weight_per_fpass": None,
             "dynamic_loss_weight_per_fpass_constant_parameter": None,

             "dynamic_loss_weight_per_tstamp": None,
             "dynamic_loss_weight_per_tstamp_constant_parameter": None,

             "dynamic_loss_weight_per_fpass_reversed": None,

             "initialise_optimiser": None,
             "optimiser_type": None,
             "sheduler_type": None,
             "cos_anneal_T_max": None,
             "learning_rate": None,
             "min_learning_rate": None,

             "sheduler_step": None,
             "sheduler_gamma":None,
             "sheduler_change": None,

             "weight_decay": None,
             "input_sampler_type": None,
             "input_sampler_type_dt": None,
             "output_sampler_type": None
             },

            ],



    )


    args = parser.parse_args()


    return args