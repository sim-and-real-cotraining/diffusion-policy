import os
import shutil
import tempfile
import subprocess
import json
from datetime import datetime
import yaml

def train_binary_classifiers(config_dir, config_name, modifications_list):
    val_accuracies = {}  # Dictionary to store val_accuracy for each modification

    for modifications in modifications_list:
        # Create a temporary directory to store the modified config
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_path = os.path.join(temp_dir, 'temp_embedding_classification.yaml')
            original_config_path = os.path.join(config_dir, config_name)
            
            # Copy the original config to the temporary file
            shutil.copy(original_config_path, temp_config_path)

            # Apply modifications to the temporary config file
            with open(temp_config_path, 'r+') as config_file:
                config_content = yaml.safe_load(config_file)
                for key, value in modifications.items():
                    keys = key.split('.')
                    d = config_content
                    for k in keys[:-1]:
                        d = d.setdefault(k, {})
                    d[keys[-1]] = value
                config_file.seek(0)
                yaml.dump(config_content, config_file)
                config_file.truncate()

            # Generate the hydra.run.dir path
            timestamp = datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
            hydra_run_dir = os.path.join('outputs/binary_classification', timestamp)

            # Split the temp config path into directory and filename
            temp_config_dir, temp_config_name = os.path.split(temp_config_path)

            # Run the training command with the modified config
            command = [
                'python', 'train.py',
                '--config-dir', temp_config_dir,
                '--config-name', temp_config_name,
                'hydra.run.dir=' + hydra_run_dir
            ]
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command, check=True)

            # Store hydra run dir for further use
            print(f"Hydra run directory: {hydra_run_dir}")

            # Read the last line of the log file and extract val_accuracy
            log_file_path = os.path.join(hydra_run_dir, 'logs.json.txt')
            max_val_accuracy = None
            try:
                with open(log_file_path, 'r') as log_file:
                    lines = log_file.readlines()
                    for line in lines:
                        log_dict = json.loads(line)
                        val_accuracy = log_dict.get('val_accuracy', None)
                        if val_accuracy is not None:
                            if max_val_accuracy is None or val_accuracy > max_val_accuracy:
                                max_val_accuracy = val_accuracy
                if max_val_accuracy is not None:
                    val_accuracies[str(modifications)] = max_val_accuracy
                else:
                    val_accuracies[str(modifications)] = float('nan')
            except FileNotFoundError:
                print(f"Log file not found: {log_file_path}")
                val_accuracies[str(modifications)] = float('nan')

    # Print the dictionary mapping modifications to val_accuracy
    print("\nValidation Accuracies:")
    for mods, accuracy in val_accuracies.items():
        print(f"{mods}: {accuracy}")

# Example usage
config_dir = 'config/binary_classification'
config_name = 'embedding_classification.yaml'
modifications_list = [
    {
        'dataset.data_0_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_3_1_emb_10/seed_0/real_embeddings.npy', 
        'dataset.data_1_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_3_1_emb_10/seed_0/sim_embeddings.npy',
        'training.num_epochs': 20
    },
    {
        'dataset.data_0_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_1_1_emb_10/seed_0/real_embeddings.npy', 
        'dataset.data_1_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_1_1_emb_10/seed_0/sim_embeddings.npy',
        'training.num_epochs': 20
    },
    {
        'dataset.data_0_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_1_3_emb_10/seed_0/real_embeddings.npy', 
        'dataset.data_1_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_1_3_emb_10/seed_0/sim_embeddings.npy',
        'training.num_epochs': 20
    },
    {
        'dataset.data_0_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_emb_10/seed_0/real_embeddings.npy', 
        'dataset.data_1_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_emb_10/seed_0/sim_embeddings.npy',
        'training.num_epochs': 20
    },
    {
        'dataset.data_0_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_3_1_emb_10/seed_0/real_embeddings.npy', 
        'dataset.data_1_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_3_1_emb_10/seed_0/sim_embeddings.npy',
        'training.num_epochs': 20
    },
    {
        'dataset.data_0_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_1_1_emb_10/seed_0/real_embeddings.npy', 
        'dataset.data_1_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_1_1_emb_10/seed_0/sim_embeddings.npy',
        'training.num_epochs': 20
    },
    {
        'dataset.data_0_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_1_3_emb_10/seed_0/real_embeddings.npy', 
        'dataset.data_1_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_1_3_emb_10/seed_0/sim_embeddings.npy',
        'training.num_epochs': 20
    },
    {
        'dataset.data_0_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_emb_10/seed_0/real_embeddings.npy', 
        'dataset.data_1_path': '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_emb_10/seed_0/sim_embeddings.npy',
        'training.num_epochs': 20
    },
]

train_binary_classifiers(config_dir, config_name, modifications_list)
