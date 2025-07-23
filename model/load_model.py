import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# import sys, os

# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))


from constant_autoregression.util import Printer, initialize_weights_xavier_uniform
from constant_autoregression.model.FNO_1d import FNO_standard_1D, FNO_standard_1D_KS1, FNO1d_t_concat, FNO1d_t_attention, FNO_standard_1D_decompose
from constant_autoregression.model.FFNO_1d import F_FNO_1D
from constant_autoregression.model.UNO import UNO_1D
from constant_autoregression.model.LSM import LSM_1D
from constant_autoregression.model.U_NET import U_NET_1D
from constant_autoregression.model.modern_UNET import modern_UNET_1D
from constant_autoregression.model.vcnef_1d import VCNeFModel

from constant_autoregression.model.fourier_transformer import MultiHeadedAttention, PositionwiseFeedForward, SpectralConv1d_wave, Embedding, Lifting, Projecting, Generator, Encoder_wave, EncoderDecoder_wave, EncoderLayer



p = Printer(n_digits=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def load_model(args, model_dict, device, multi_gpu=False, **kwargs):


    if args.model_type == "FNO_standard_1D":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1") or args.dataset_name.endswith("B1") or args.dataset_name.endswith("A1"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps }
            elif args.dataset_name.endswith("KS1") or args.dataset_name.endswith("KdV"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps+args.no_parameters, "output_size": args.output_time_stamps }
        
        elif args.time_prediction.startswith("variable"):
            if args.time_conditioning.startswith("addition"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps }
            elif args.time_conditioning.startswith("concatenate"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": 2*args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps }        
        else:
            raise TypeError("Specify the input shape type")

        if args.dataset_name.endswith("E1")  or args.dataset_name.endswith("B1") or args.dataset_name.endswith("A1") or args.dataset_name.endswith("KdV"):
            model = FNO_standard_1D(
                model_hyperparameters["modes"],
                model_hyperparameters["width"],
                model_hyperparameters["input_size"],
                model_hyperparameters["output_size"]
                ).to(device)
            
        elif args.dataset_name.endswith("KS1"):
            model = FNO_standard_1D_KS1(
                model_hyperparameters["modes"],
                model_hyperparameters["width"],
                model_hyperparameters["input_size"],
                model_hyperparameters["output_size"]
                ).to(device)

    elif args.model_type == "FNO_standard_1D_decompose":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps }
            elif args.dataset_name.endswith("E2"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps + 3, "output_size": args.output_time_stamps }
        model = FNO_standard_1D_decompose(
            model_hyperparameters["modes"],
            model_hyperparameters["width"],
            model_hyperparameters["input_size"],
            model_hyperparameters["output_size"],

            ).to(device)
        
    elif args.model_type == "F_FNO_1D":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1")  or args.dataset_name.endswith("B1") :        
                model_hyperparameters = {
                    "modes": args.fno_modes,
                    "width": args.fno_hidden_dim,
                    "input_dim": args.input_time_stamps,
                    "output_dim": args.output_time_stamps,
                    "dropout": 0.0,
                    "in_dropout": 0.0,
                    "n_layers": args.fno_hidden_layers,
                    "share_weight": True,
                    
                    "factor": 2,
                    "ff_weight_norm": True,
                    "n_ff_layers": 2,
                    "gain": 1,
                    "layer_norm": False,
                    
                    "share_fork": False,
                    "use_fork": False,
                    "mode": 'full'
                    }
        
        model = F_FNO_1D(
                    model_hyperparameters["modes"],
                    model_hyperparameters["width"],
                    model_hyperparameters["input_dim"],
                    model_hyperparameters["output_dim"],
                    model_hyperparameters["dropout"],
                    model_hyperparameters["in_dropout"],
                    model_hyperparameters["n_layers"],
                    model_hyperparameters["share_weight"],
                    
                    model_hyperparameters["factor"],
                    model_hyperparameters["ff_weight_norm"],
                    model_hyperparameters["n_ff_layers"],
                    model_hyperparameters["gain"],
                    model_hyperparameters["layer_norm"],
                    
                    model_hyperparameters["share_fork"],
                    model_hyperparameters["use_fork"],
                    model_hyperparameters["mode"]

        ).to(device)



    elif args.model_type ==  "UNO_1D":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1")  or args.dataset_name.endswith("B1") or args.dataset_name.endswith("A1"):
                model_hyperparameters = {"in_width": args.input_time_stamps, "out_dim": args.output_time_stamps, "width": args.fno_hidden_dim}
            if args.dataset_name.endswith("KdV"):
                model_hyperparameters = {"in_width": args.input_time_stamps + 2 , "out_dim": args.output_time_stamps, "width": args.fno_hidden_dim}
        
        model = UNO_1D(
            model_hyperparameters["in_width"],
            model_hyperparameters["out_dim"],
            model_hyperparameters["width"]
        ).to(device)



    elif args.model_type ==  "LSM_1D":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1") or args.dataset_name.endswith("B1") or args.dataset_name.endswith("A1"):
                model_hyperparameters = {"in_dim": args.input_time_stamps, "out_dim": args.output_time_stamps, "d_model": 16, "num_token": 4, "num_basis": 12, "patch_size": "4", "padding": "32"}
                #model_hyperparameters = {"in_dim": 1, "out_dim": 1, "d_model": 32, "num_token": 4, "num_basis": 24, "patch_size": "8", "padding": "14"}
            if args.dataset_name.endswith("KdV"):
                model_hyperparameters = {"in_dim": args.input_time_stamps + 2, "out_dim": args.output_time_stamps, "d_model": 16, "num_token": 4, "num_basis": 12, "patch_size": "4", "padding": "28"}
        model = LSM_1D(
            model_hyperparameters["in_dim"],
            model_hyperparameters["out_dim"],
            model_hyperparameters["d_model"],
            model_hyperparameters["num_token"],
            model_hyperparameters["num_basis"],
            model_hyperparameters["patch_size"],
            model_hyperparameters["padding"]
        ).to(device)


    elif args.model_type ==  "U_NET_1D":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1")  or args.dataset_name.endswith("B1") or args.dataset_name.endswith("A1") or args.dataset_name.endswith("KdV"):
                #model_hyperparameters = {"n_input_scalar_components": 1, "n_input_vector_components": 0, "n_output_scalar_components": 1, "n_output_vector_components": 0, "time_history": args.input_time_stamps, "time_future": args.output_time_stamps, "hidden_channels": args.fno_hidden_dim, "padding": 32}
                model_hyperparameters = {"n_input_scalar_components": 1, "n_input_vector_components": 0, "n_output_scalar_components": 1, "n_output_vector_components": 0, "time_history": args.input_time_stamps, "time_future": args.output_time_stamps, "hidden_channels": 16 , "padding": 32}
            elif args.dataset_name.endswith("KdV"):
                #model_hyperparameters = {"n_input_scalar_components": 1, "n_input_vector_components": 0, "n_output_scalar_components": 1, "n_output_vector_components": 0, "time_history": args.input_time_stamps + 2, "time_future": args.output_time_stamps, "hidden_channels": args.fno_hidden_dim, "padding": 32}
                model_hyperparameters = {"n_input_scalar_components": 1, "n_input_vector_components": 0, "n_output_scalar_components": 1, "n_output_vector_components": 0, "time_history": args.input_time_stamps + 2, "time_future": args.output_time_stamps, "hidden_channels": 16, "padding": 32}
        model = U_NET_1D(
                model_hyperparameters["n_input_scalar_components"],
                model_hyperparameters["n_input_vector_components"],
                model_hyperparameters["n_output_scalar_components"],
                model_hyperparameters["n_output_vector_components"],
                model_hyperparameters["time_history"],
                model_hyperparameters["time_future"],
                model_hyperparameters["hidden_channels"],
                model_hyperparameters["padding"]
            ).to(device)
        

    elif args.model_type ==  "modern_UNET_1D":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1")  or args.dataset_name.endswith("B1") or args.dataset_name.endswith("A1") or args.dataset_name.endswith("KdV"):
                model_hyperparameters = {"n_input_scalar_components": 1, "n_input_vector_components": 0, "n_output_scalar_components": 1, "n_output_vector_components": 0, "time_history":   args.input_time_stamps, "time_future": args.output_time_stamps, "hidden_channels": args.fno_hidden_dim, "activation": nn.GELU(), "norm": False, "ch_mults": (1,2,3,4), "is_attn":(False, False, False, False), "mid_attn": False, "n_blocks": 1, "use1x1": True, "padding": 8}

        model = modern_UNET_1D(
                model_hyperparameters["n_input_scalar_components"],
                model_hyperparameters["n_input_vector_components"],
                model_hyperparameters["n_output_scalar_components"],
                model_hyperparameters["n_output_vector_components"],
                model_hyperparameters["time_history"],
                model_hyperparameters["time_future"],
                model_hyperparameters["hidden_channels"],
                model_hyperparameters["activation"],
                model_hyperparameters["norm"],
                model_hyperparameters["ch_mults"],
                model_hyperparameters["is_attn"],
                model_hyperparameters["mid_attn"],
                model_hyperparameters["n_blocks"],
                model_hyperparameters["use1x1"],
                model_hyperparameters["padding"]
            ).to(device)
        

    elif args.model_type == "FNO1d_t_concat":
        assert args.time_prediction.startswith("variable") 
        if args.time_conditioning.startswith("concatenate"):
            model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": 2*args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps, "x_res": args.x_resolution }   
        
        model = FNO1d_t_concat(
            model_hyperparameters["modes"],
            model_hyperparameters["width"],
            model_hyperparameters["input_size"],
            model_hyperparameters["output_size"],
            model_hyperparameters["x_res"],

            ).to(device)

    
    
    elif args.model_type == "FNO1d_t_attention":
        assert args.time_prediction.startswith("variable") 
        if args.time_conditioning.startswith("attention"):
            model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps, "t_res": args.t_resolution, "x_res": args.x_resolution, "nhead": 1 }
        
        model = FNO1d_t_attention(
            model_hyperparameters["modes"],
            model_hyperparameters["width"],
            model_hyperparameters["input_size"],
            model_hyperparameters["output_size"],
            model_hyperparameters["t_res"],
            model_hyperparameters["x_res"],
            model_hyperparameters["nhead"]

            ).to(device)


    elif args.model_type == "vcnef_1d":
        #assert args.time_prediction.startswith("variable") 

        #model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps, "t_res": args.t_resolution, "x_res": args.x_resolution, "nhead": 1 }
        model_hyperparameters = {"num_channels": args.input_time_stamps, "condition_on_pde_param": False, "pde_param_dim": 1, "d_model": args.fno_hidden_dim, "n_heads": 8, "n_transformer_blocks": 3, "n_modulation_blocks": 3}
                
        model = VCNeFModel(
            num_channels = model_hyperparameters["num_channels"],
            condition_on_pde_param = model_hyperparameters["condition_on_pde_param"],
            pde_param_dim = model_hyperparameters["pde_param_dim"],
            d_model = model_hyperparameters["d_model"],
            n_heads = model_hyperparameters["n_heads"],
            n_transformer_blocks = model_hyperparameters["n_transformer_blocks"],
            n_modulation_blocks = model_hyperparameters["n_modulation_blocks"]

            ).to(device)
        

        
    elif args.model_type == "Fourier_transformer":
        assert args.time_prediction.startswith("variable")
        modes_in = model_hyperparameters["modes"]
        modes_out = model_hyperparameters["modes"]
        d_model =  model_hyperparameters["width"]
        N =  model_hyperparameters["layers"]
        time_seq = model_hyperparameters["input_size"]
        d_ff = model_hyperparameters["width"]*2
        dropout = 0.1
        "Helper: Construct a model from hyperparameters."
        attn = MultiHeadedAttention(h, d_model, modes_in)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        spec_conv_wave =  SpectralConv1d_wave(modes_in, modes_out, time_seq, d_model)
        embeddings = Embedding(d_model)

        # subconnection = N*[ ["norm_residual","norm_residual","_activation"] ]
        # subconnection[-1] = ["norm_residual","norm_residual","_"]

        subconnection = N*[ ["norm_residual","_activation"] ]
        subconnection[-1] = ["norm_residual","_"]

        model = EncoderDecoder_wave(
            Embedding(d_model),
            Encoder_wave(EncoderLayer(attn, ff, spec_conv_wave, dropout), N, subconnection, embeddings),
            #Decoder(DecoderLayer(attn, attn, ff, spec_conv, dropout), N),
            #Generator(d_model),
            Lifting(d_model),
            Projecting(d_model),

    )
    p.print("Loading Model last state")
    #model.load_state_dict(model_dict["state_dict"])
    model.load_state_dict(model_dict)
    model.to(device)
    return model






def get_model(args, device, **kwargs):
    
    if args.model_type == "FNO_standard_1D":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1") or args.dataset_name.endswith("B1") or args.dataset_name.endswith("A1"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps }
            elif args.dataset_name.endswith("E2"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps + args.no_parameters, "output_size": args.output_time_stamps }
            elif args.dataset_name.endswith("KS1") or args.dataset_name.endswith("KdV"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps + args.no_parameters, "output_size": args.output_time_stamps }

        elif args.time_prediction.startswith("variable"):
            if args.time_conditioning.startswith("addition"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps }
            elif args.time_conditioning.startswith("concatenate"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": 2*args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps }
        else:
            raise TypeError("Specify the input shape type")

        if args.dataset_name.endswith("E1") or args.dataset_name.endswith("B1") or args.dataset_name.endswith("A1") or args.dataset_name.endswith("KdV"):
            model = FNO_standard_1D(
                model_hyperparameters["modes"],
                model_hyperparameters["width"],
                model_hyperparameters["input_size"],
                model_hyperparameters["output_size"]
                ).to(device)
            
        elif args.dataset_name.endswith("KS1"):
            model = FNO_standard_1D_KS1(
                model_hyperparameters["modes"],
                model_hyperparameters["width"],
                model_hyperparameters["input_size"],
                model_hyperparameters["output_size"]
                ).to(device)
            

    elif args.model_type == "FNO_standard_1D_decompose":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps }
            elif args.dataset_name.endswith("E2"):
                model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps + 3, "output_size": args.output_time_stamps }
    
        model = FNO_standard_1D_decompose(
            model_hyperparameters["modes"],
            model_hyperparameters["width"],
            model_hyperparameters["input_size"],
            model_hyperparameters["output_size"],
            ).to(device)


    elif args.model_type == "F_FNO_1D":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1")  or args.dataset_name.endswith("B1") :        
                model_hyperparameters = {
                    "modes": args.fno_modes,
                    "width": args.fno_hidden_dim,
                    "input_dim": args.input_time_stamps,
                    "output_dim": args.output_time_stamps,
                    "dropout": 0.0,
                    "in_dropout": 0.0,
                    "n_layers": args.fno_hidden_layers,
                    "share_weight": True,
                    
                    "factor": 2,
                    "ff_weight_norm": True,
                    "n_ff_layers": 2,
                    "gain": 1,
                    "layer_norm": False,
                    
                    "share_fork": False,
                    "use_fork": False,
                    "mode": 'full'
                    }
        
        model = F_FNO_1D(
                    model_hyperparameters["modes"],
                    model_hyperparameters["width"],
                    model_hyperparameters["input_dim"],
                    model_hyperparameters["output_dim"],
                    model_hyperparameters["dropout"],
                    model_hyperparameters["in_dropout"],
                    model_hyperparameters["n_layers"],
                    model_hyperparameters["share_weight"],
                    
                    model_hyperparameters["factor"],
                    model_hyperparameters["ff_weight_norm"],
                    model_hyperparameters["n_ff_layers"],
                    model_hyperparameters["gain"],
                    model_hyperparameters["layer_norm"],
                    
                    model_hyperparameters["share_fork"],
                    model_hyperparameters["use_fork"],
                    model_hyperparameters["mode"]

        ).to(device)
    
    
    

    elif args.model_type ==  "UNO_1D":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1")  or args.dataset_name.endswith("B1") or args.dataset_name.endswith("A1"):
                model_hyperparameters = {"in_width": args.input_time_stamps, "out_dim": args.output_time_stamps, "width": args.fno_hidden_dim}   
            if args.dataset_name.endswith("KdV"):
                model_hyperparameters = {"in_width": args.input_time_stamps + 2 , "out_dim": args.output_time_stamps, "width": args.fno_hidden_dim}        
        model = UNO_1D(
            model_hyperparameters["in_width"],
            model_hyperparameters["out_dim"],
            model_hyperparameters["width"]
        ).to(device)



    elif args.model_type ==  "LSM_1D":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1") or args.dataset_name.endswith("B1") or args.dataset_name.endswith("A1"):
                model_hyperparameters = {"in_dim": args.input_time_stamps, "out_dim": args.output_time_stamps, "d_model": 16, "num_token": 4, "num_basis": 12, "patch_size": "4", "padding": "32"}
                #model_hyperparameters = {"in_dim": 1, "out_dim": 1, "d_model": 32, "num_token": 4, "num_basis": 24, "patch_size": "8", "padding": "14"}
            if args.dataset_name.endswith("KdV"):
                model_hyperparameters = {"in_dim": args.input_time_stamps + 2, "out_dim": args.output_time_stamps, "d_model": 16, "num_token": 4, "num_basis": 12, "patch_size": "4", "padding": "28"}
        model = LSM_1D(
            model_hyperparameters["in_dim"],
            model_hyperparameters["out_dim"],
            model_hyperparameters["d_model"],
            model_hyperparameters["num_token"],
            model_hyperparameters["num_basis"],
            model_hyperparameters["patch_size"],
            model_hyperparameters["padding"]
        ).to(device)



    elif args.model_type ==  "U_NET_1D":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1")  or args.dataset_name.endswith("B1") or args.dataset_name.endswith("A1") or args.dataset_name.endswith("KdV"):
                #model_hyperparameters = {"n_input_scalar_components": 1, "n_input_vector_components": 0, "n_output_scalar_components": 1, "n_output_vector_components": 0, "time_history": args.input_time_stamps, "time_future": args.output_time_stamps, "hidden_channels": args.fno_hidden_dim, "padding": 32}
                model_hyperparameters = {"n_input_scalar_components": 1, "n_input_vector_components": 0, "n_output_scalar_components": 1, "n_output_vector_components": 0, "time_history": args.input_time_stamps, "time_future": args.output_time_stamps, "hidden_channels": 16 , "padding": 32}
            elif args.dataset_name.endswith("KdV"):
                #model_hyperparameters = {"n_input_scalar_components": 1, "n_input_vector_components": 0, "n_output_scalar_components": 1, "n_output_vector_components": 0, "time_history": args.input_time_stamps + 2, "time_future": args.output_time_stamps, "hidden_channels": args.fno_hidden_dim, "padding": 32}
                model_hyperparameters = {"n_input_scalar_components": 1, "n_input_vector_components": 0, "n_output_scalar_components": 1, "n_output_vector_components": 0, "time_history": args.input_time_stamps + 2, "time_future": args.output_time_stamps, "hidden_channels": 16, "padding": 32}
        model = U_NET_1D(
                model_hyperparameters["n_input_scalar_components"],
                model_hyperparameters["n_input_vector_components"],
                model_hyperparameters["n_output_scalar_components"],
                model_hyperparameters["n_output_vector_components"],
                model_hyperparameters["time_history"],
                model_hyperparameters["time_future"],
                model_hyperparameters["hidden_channels"],
                model_hyperparameters["padding"]
            ).to(device)


    elif args.model_type ==  "modern_UNET_1D":
        if args.time_prediction.startswith("constant"):
            if args.dataset_name.endswith("E1")  or args.dataset_name.endswith("B1") or args.dataset_name.endswith("A1") or args.dataset_name.endswith("KdV"):
                model_hyperparameters = {"n_input_scalar_components": 1, "n_input_vector_components": 0, "n_output_scalar_components": 1, "n_output_vector_components": 0, "time_history": args.input_time_stamps, "time_future": args.output_time_stamps, "hidden_channels": args.fno_hidden_dim, "activation": nn.GELU(), "norm": False, "ch_mults": (1,2,3,4), "is_attn":(False, False, False, False), "mid_attn": False, "n_blocks": 1, "use1x1": True, "padding": 8}
        model = modern_UNET_1D(
                model_hyperparameters["n_input_scalar_components"],
                model_hyperparameters["n_input_vector_components"],
                model_hyperparameters["n_output_scalar_components"],
                model_hyperparameters["n_output_vector_components"],
                model_hyperparameters["time_history"],
                model_hyperparameters["time_future"],
                model_hyperparameters["hidden_channels"],
                model_hyperparameters["activation"],
                model_hyperparameters["norm"],
                model_hyperparameters["ch_mults"],
                model_hyperparameters["is_attn"],
                model_hyperparameters["mid_attn"],
                model_hyperparameters["n_blocks"],
                model_hyperparameters["use1x1"],
                model_hyperparameters["padding"]
            ).to(device)
        

    elif args.model_type == "FNO1d_t_concat":
        assert args.time_prediction.startswith("variable")
        if args.time_conditioning.startswith("concatenate"):
            model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": 2*args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps, "x_res": args.x_resolution }   
        
        model = FNO1d_t_concat(
            model_hyperparameters["modes"],
            model_hyperparameters["width"],
            model_hyperparameters["input_size"],
            model_hyperparameters["output_size"],
            model_hyperparameters["x_res"],

            ).to(device)
        

    elif args.model_type == "FNO1d_t_attention":
        assert args.time_prediction.startswith("variable")
        if args.time_conditioning.startswith("attention"):
            model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps, "t_res": args.t_resolution, "x_res": args.x_resolution, "nhead": 1 }
        model = FNO1d_t_attention(
            model_hyperparameters["modes"],
            model_hyperparameters["width"],
            model_hyperparameters["input_size"],
            model_hyperparameters["output_size"],
            model_hyperparameters["t_res"],
            model_hyperparameters["x_res"],
            model_hyperparameters["nhead"]

            ).to(device)


    elif args.model_type == "Fourier_transformer":
        assert args.time_prediction.startswith("variable")
        model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps, "layers": args.fno_hidden_layers, "x_res": args.x_resolution }
        modes_in = model_hyperparameters["modes"]
        modes_out = model_hyperparameters["modes"]
        d_model =  model_hyperparameters["width"]
        N =  model_hyperparameters["layers"]
        time_seq = model_hyperparameters["input_size"]
        d_ff = model_hyperparameters["width"]*2
        dropout = 0.1
        h = 1
        "Helper: Construct a model from hyperparameters."
        attn = MultiHeadedAttention(h, d_model, modes_in)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        spec_conv_wave =  SpectralConv1d_wave(modes_in, modes_out, time_seq, d_model)
        embeddings = Embedding(d_model)

        # subconnection = N*[ ["norm_residual","norm_residual","_activation"] ]
        # subconnection[-1] = ["norm_residual","norm_residual","_"]

        subconnection = N*[ ["norm_residual","_activation"] ]
        subconnection[-1] = ["norm_residual","_"]

        
        model = EncoderDecoder_wave(
            Embedding(d_model),
            Encoder_wave(EncoderLayer(attn, ff, spec_conv_wave, dropout), N, subconnection, embeddings),
            #Decoder(DecoderLayer(attn, attn, ff, spec_conv, dropout), N),
            Generator(d_model),
            Lifting(d_model),
            Projecting(d_model),

    ).to(device)




    elif args.model_type == "vcnef_1d":
        #assert args.time_prediction.startswith("variable") 

        #model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps, "t_res": args.t_resolution, "x_res": args.x_resolution, "nhead": 1 }
        #model_hyperparameters = {"num_channels": args.input_time_stamps, "condition_on_pde_param": False, "pde_param_dim": 1, "d_model": args.fno_hidden_dim, "n_heads": 4, "n_transformer_blocks": 3, "n_modulation_blocks": 3}
        model_hyperparameters = {"num_channels": args.input_time_stamps, "condition_on_pde_param": False, "pde_param_dim": 1, "d_model": args.fno_hidden_dim, "n_heads": 8, "n_transformer_blocks": 3, "n_modulation_blocks": 3}      
        model = VCNeFModel(
            num_channels = model_hyperparameters["num_channels"],
            condition_on_pde_param = model_hyperparameters["condition_on_pde_param"],
            pde_param_dim = model_hyperparameters["pde_param_dim"],
            d_model = model_hyperparameters["d_model"],
            n_heads = model_hyperparameters["n_heads"],
            n_transformer_blocks = model_hyperparameters["n_transformer_blocks"],
            n_modulation_blocks = model_hyperparameters["n_modulation_blocks"]

            ).to(device)
        
        
    else:
        raise TypeError(f"model type: {args.model_type} does not exist") 
    
    if args.model_initialise_type == "xavier_uniform":
        initialize_weights_xavier_uniform(model)
        p.print("Model with xavier_uniform")
    else:
        p.print("Model with Random Intitialisation")

    return model

    