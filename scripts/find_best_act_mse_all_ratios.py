import os
import json
import argparse
from glob import glob

def find_best_val_ddim_mse_in_prefix(prefix_path):
    """
    Find the lowest val_ddim_mse_0 among all wandb runs that match the given directory prefix.

    Args:
        prefix_path (str): The directory prefix to search for wandb run directories.

    Prints:
        - The directory with the lowest val_ddim_mse_0
        - The lowest val_ddim_mse_0 value
        - The epoch where it occurred
        - The total number of val_ddim_mse_0 measurements across all directories
        - The total number of unique epochs across all directories
    """
    # Find all wandb runs that match the prefix
    wandb_run_dirs = [d for d in glob(f"{prefix_path}*", recursive=False) if os.path.isdir(d)]
    
    if not wandb_run_dirs:
        print(f"No matching directories found for prefix: {prefix_path}")
        return

    global_min_mse = float("inf")
    best_dir = None
    best_epoch = None
    total_mse_count = 0
    total_epochs_seen = set()

    for wandb_run_dir in wandb_run_dirs:
        log_file = os.path.join(wandb_run_dir, "logs.json.txt")
        if not os.path.exists(log_file):
            continue

        min_mse = float("inf")
        best_epoch_local = None
        mse_count = 0
        epochs_seen = set()

        with open(log_file, "r") as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    if "epoch" in log_entry:
                        epochs_seen.add(log_entry["epoch"])
                    if "val_ddim_mse_0" in log_entry:
                        mse_value = log_entry["val_ddim_mse_0"]
                        mse_count += 1
                        if mse_value < min_mse:
                            min_mse = mse_value
                            best_epoch_local = log_entry.get("epoch", "Unknown")
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in {wandb_run_dir}: {line.strip()}")

        # Update global best if this directory has a lower val_ddim_mse_0
        if min_mse < global_min_mse:
            global_min_mse = min_mse
            best_epoch = best_epoch_local
            best_dir = wandb_run_dir

        total_mse_count += mse_count
        total_epochs_seen.update(epochs_seen)

    total_epochs = len(total_epochs_seen)

    if best_dir:
        print(f"Best directory: {best_dir}")
        print(f"Lowest val_ddim_mse_0: {global_min_mse:.6f} at epoch {best_epoch}")
        print(f"Total val_ddim_mse_0 measurements: {total_mse_count}")
        print(f"Total unique epochs across all directories: {total_epochs}")
    else:
        print("No val_ddim_mse_0 measurements found in any directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the lowest val_ddim_mse_0 among multiple wandb runs.")
    parser.add_argument("prefix_path", type=str, help="Prefix path to search for wandb run directories")
    args = parser.parse_args()

    find_best_val_ddim_mse_in_prefix(args.prefix_path)
