import json
import os
import subprocess

# --- Configuration ---
json_file_path = "/mnt/scratch/scoc/constant_autoregression/arguments.json"  # Replace with the actual path to your .json file
job_script_path = "/mnt/scratch/scoc/constant_autoregression/run_all.sh"    # Replace with the actual path to your job.sh file
output_dir = "/mnt/scratch/scoc/constant_autoregression/jobs_arugments"   # Replace with where you want to save modified JSON files (optional)

result_dir = "/mnt/scratch/scoc/constant_autoregression/result"   # Replace with where you want to save modified JSON files (optional)

dataset_names = ["B1",]

#analysis_types = ["FNO", ]  # Your list of analysis types
analysis_types = ["model_types", ]  # Your list of analysis types

model_types = ["FNO_standard_1D", ]

subanalysis_types = [
                    "AR",
                    "AR_curriculum",
                    # "TF",
                    # "TF_noise",
                    # "AR_TF_curriculum",
                    # "AR_TF_noise_curriculum",
                    # "TF_AR_Prob",
                    # "AR_TF_Prob",
                    "STWL_curriculum"
                      ]  # Your list of subanalysis types
training_loops = [
                    "fixed_autoregressive_rollout",
                    "autoregressive_rollout_curriculum",
                    # "teacher_forcing",
                    # "teacher_forcing_with_noise",
                    # "autoregressive_rollout_to_teacher_forcing",
                    # "autoregressive_rollout_to_teacher_forcing_with_noise",
                    # "probabilistic_teacher_forcing_to_autoregressive_rollout",
                    # "probabilistic_autoregressive_rollout_to_teacher_forcing",
                    "scheduled_weighted_loss_curriculum"
                      ]  # Your list of subanalysis types


# subanalysis_types = [
#                     #"AR_curriculum",
#                     # "TF",
#                     # "TF_noise",
#                     #"AR_TF_curriculum",
#                     #"TF_AR_Prob",
#                     #"AR_TF_Prob",
#                       ]  # Your list of subanalysis types
# training_loops = [
#                     # "autoregressive_rollout_curriculum",
#                     # "teacher_forcing",
#                     # "teacher_forcing_with_noise",
#                     #"autoregressive_rollout_to_teacher_forcing",
#                     #"probabilistic_teacher_forcing_to_autoregressive_rollout",
#                     #"probabilistic_autoregressive_rollout_to_teacher_forcing",
#                       ]  # Your list of subanalysis types






#experiments = ["run_1", "run_2", "run_3", "run_4", "run_5"]  # Your list of experiments
#experiments = ["run_1", ]  # Your list of experiments


#experiments = ["run_B_128_H_1_I_256_UNO_1D",]  # Your list of experiments

seeds = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]

models_list = ["U_NET_1D"]

experiments = ["run_B_128_H_32_I_8_" + i +"_small" for i in models_list]  # Your list of experiments

# --- Main Logic ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("output_dir", output_dir)


for dataset_name in dataset_names:
    for analysis_type in analysis_types:
        for subanalysis_type in subanalysis_types:
            for experiment in experiments:
                print("\n")
                # 1. Load the JSON file
                try:
                    with open(json_file_path, 'r') as f:
                        arguments = json.load(f)
                except FileNotFoundError:
                    print(f"Error: JSON file not found at {json_file_path}")
                    continue
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from {json_file_path}")
                    continue

                # 2. Modify the relevant arguments
                arguments["dataset_name"] = dataset_name

                arguments["analysis_type"] = analysis_type
                arguments["model_type"] = model_types[analysis_types.index(analysis_type)]

                arguments["subanalysis_type"] = subanalysis_type
                arguments["training_loop"] = training_loops[subanalysis_types.index(subanalysis_type)]

                arguments["experiment"] = experiment
                arguments["seed"] = seeds[experiments.index(experiment)]


#########################################################################################
                arguments["model_type"] = models_list[experiments.index(experiment)]
                arguments["batch_size_train"] = 128

                arguments["training_protocols"][0]["epochs"] = 200
                arguments["training_protocols"][0]["iter_per_epochs"] = 8
                arguments["training_protocols"][0]["horizon"] = [32,]


                arguments["training_protocols"][0]["sheduler_gamma"] = 0.98
                arguments["training_protocols"][0]["sheduler_step"] = 1

#########################################################################################


                # 3. Save the modified JSON file (optional)
                output_json_filename = f"argument_{dataset_name}_{analysis_type}_{subanalysis_type}_{experiment}"
                output_json_path = os.path.join(output_dir, output_json_filename)
                try:
                    with open(output_json_path +".json", 'w') as f:
                        json.dump(arguments, f, indent=4)  # indent for readability
                    print(f"Modified JSON saved to: {output_json_path}")
                except Exception as e:
                    print(f"Error saving modified JSON: {e}")
                    continue
                
                result_loc = os.path.join(result_dir, dataset_name, analysis_type, subanalysis_type, experiment)
                try:
                    with open(output_json_path +".txt", 'w') as f:
                        f.write(result_loc)  # indent for readability
                    print(f" result location saved to: {output_json_path}")
                except Exception as e:
                    print(f"Error saving result location txt: {e}")
                    continue

                # 4. Submit the job using sbatch
                try:
                    command = ["sbatch", job_script_path, output_json_path] # Pass the modified JSON path as an argument
                    result = subprocess.run(command, capture_output=True, text=True, check=True)
                    print(f"Job submitted for: analysis={analysis_type}, subanalysis={subanalysis_type}, experiment={experiment}")
                    print(f"Slurm output: {result.stdout.strip()}")
                    if result.stderr:
                        print(f"Slurm error: {result.stderr.strip()}")
                except FileNotFoundError:
                    print(f"Error: Job script not found at {job_script_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error submitting job: {e}")
                    print(f"Stderr: {e.stderr}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

print("Finished iterating through all configurations and submitting jobs.")