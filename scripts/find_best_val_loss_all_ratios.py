import os
import json
import argparse
from glob import glob

def find_best_val_loss_in_prefix(prefix_path):
    """
    Find the lowest val_loss_0 among all wandb runs that match the given directory prefix.

    Args:
        prefix_path (str): The directory prefix to search for wandb run directories.

    Prints:
        - The directory with the lowest val_loss_0
        - The lowest val_loss_0 value
        - The epoch where it occurred
        - The total number of val_loss_0 measurements across all directories
        - The total number of unique epochs across all directories
    """
    # Find all wandb runs that match the prefix
    wandb_run_dirs = [d for d in glob(f"{prefix_path}*", recursive=False) if os.path.isdir(d)]
    
    if not wandb_run_dirs:
        print(f"No matching directories found for prefix: {prefix_path}")
        return

    global_min_val_loss = float("inf")
    best_dir = None
    best_epoch = None
    total_val_loss_count = 0
    total_epochs_seen = set()

    for wandb_run_dir in wandb_run_dirs:
        log_file = os.path.join(wandb_run_dir, "logs.json.txt")
        if not os.path.exists(log_file):
            continue

        min_val_loss = float("inf")
        best_epoch_local = None
        val_loss_count = 0
        epochs_seen = set()

        with open(log_file, "r") as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    if "epoch" in log_entry:
                        epochs_seen.add(log_entry["epoch"])
                    if "val_loss_0" in log_entry:
                        val_loss_0 = log_entry["val_loss_0"]
                        val_loss_count += 1
                        if val_loss_0 < min_val_loss:
                            min_val_loss = val_loss_0
                            best_epoch_local = log_entry.get("epoch", "Unknown")
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in {wandb_run_dir}: {line.strip()}")

        # Update global best if this directory has a lower val_loss_0
        if min_val_loss < global_min_val_loss:
            global_min_val_loss = min_val_loss
            best_epoch = best_epoch_local
            best_dir = wandb_run_dir

        total_val_loss_count += val_loss_count
        total_epochs_seen.update(epochs_seen)

    total_epochs = len(total_epochs_seen)

    if best_dir:
        print(f"Best directory: {best_dir}")
        print(f"Lowest val_loss_0: {global_min_val_loss:.6f} at epoch {best_epoch}")
        print(f"Total val_loss_0 measurements: {total_val_loss_count}")
        print(f"Total unique epochs across all directories: {total_epochs}")
    else:
        print("No val_loss_0 measurements found in any directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the lowest val_loss_0 among multiple wandb runs.")
    parser.add_argument("prefix_path", type=str, help="Prefix path to search for wandb run directories")
    args = parser.parse_args()

    find_best_val_loss_in_prefix(args.prefix_path)