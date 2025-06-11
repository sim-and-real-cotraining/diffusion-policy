import os
import fileinput
import argparse

"""
Example usage:

python submit_trainings.py --config_dir <path_to_configs> --hydra_run_dir <path to run dir parent>
"""


def main(config_dir, hydra_run_dir, num_cores, num_GPUs):
    # Loop through each config file in the config directory
    yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    yaml_files.sort()  # Sort the files for consistent processing
    for config_file in yaml_files:
        # Set config_dir, config_name, and hydra_run_dir accordingly
        config_name = config_file
        run_dir = os.path.join(hydra_run_dir, config_name.replace(".yaml", ""))

        # Modify the submit_training.sh file
        with fileinput.FileInput('submit_training.sh', inplace=True) as file:
            for line in file:
                if line.startswith("CONFIG_DIR="):
                    print(f"CONFIG_DIR={config_dir}")
                elif line.startswith("CONFIG_NAME="):
                    print(f"CONFIG_NAME={config_name}")
                elif line.startswith("HYDRA_RUN_DIR="):
                    print(f"HYDRA_RUN_DIR={run_dir}")
                else:
                    print(line, end='')
        
        # LLsub command to launch the modified training script
        command = ["LLsub", "./submit_training.sh", "-s", f"{num_cores}", "-g", f"volta:{num_GPUs}"]
        print("-------------------------------------------")
        print(f"Config Directory: {config_dir}")
        print(f"Config Name: {config_name}")
        print(f"Hydra Run Directory: {run_dir}")
        print(command)

        # Print and run the command
        os.system(' '.join(command))
        print("-------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch LLsub for each config YAML in the specified directory.")
    parser.add_argument("--config_dir", type=str, help="Path to the config directory")
    parser.add_argument("--hydra_run_dir", type=str, help="Base Hydra run directory")
    parser.add_argument("--s", type=int, help="Number of cores to use", default=20)
    parser.add_argument("--g", type=int, help="Number of GPUs to use", default=1)
    args = parser.parse_args()
    
    main(args.config_dir, args.hydra_run_dir, args.s, args.g)
