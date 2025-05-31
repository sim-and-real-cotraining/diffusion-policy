import os
import json
import argparse

def find_val_loss(wandb_run_dir, target_epoch=None):
    """
    Find the val_loss_0 for a specific epoch or the best overall val_loss_0 from logs.json.txt in a given wandb run directory.

    Args:
        wandb_run_dir (str): Path to the wandb run directory.
        target_epoch (str or int, optional): The epoch to retrieve val_loss_0 for. If "latest", returns the last recorded val_loss_0.

    Prints:
        - If target_epoch is None: The minimum val_loss_0 value, the number of val_loss_0 measurements taken, 
          the epoch where the minimum occurred, and the total number of unique epochs recorded.
        - If target_epoch is provided: The val_loss_0 for the given epoch or a message if not found.
    """
    log_file = os.path.join(wandb_run_dir, "logs.json.txt")
    
    if not os.path.exists(log_file):
        print(f"Error: logs.json.txt not found in {wandb_run_dir}")
        return
    
    min_val_loss = float("inf")
    best_epoch = None
    val_loss_count = 0
    epochs_seen = set()
    latest_val_loss = None
    latest_epoch = None
    val_loss_at_epoch = None
    
    with open(log_file, "r") as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                epoch = log_entry.get("epoch")
                
                if epoch is not None:
                    epochs_seen.add(epoch)
                
                if "val_loss_0" in log_entry:
                    val_loss_0 = log_entry["val_loss_0"]
                    val_loss_count += 1
                    
                    # Track latest val_loss_0
                    latest_val_loss = val_loss_0
                    latest_epoch = epoch
                    
                    # Track best val_loss_0
                    if val_loss_0 < min_val_loss:
                        min_val_loss = val_loss_0
                        best_epoch = epoch
                    
                    # Track specific epoch's val_loss_0
                    if target_epoch is not None and str(epoch) == str(target_epoch):
                        val_loss_at_epoch = val_loss_0
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line.strip()}")
    
    total_epochs = len(epochs_seen)
    
    if target_epoch is None:
        if val_loss_count > 0:
            print(f"Total val_loss_0 measurements: {val_loss_count}")
            print(f"Lowest val_loss_0: {min_val_loss:.6f} at epoch {best_epoch}")
            print(f"Total number of unique epochs: {total_epochs}")
        else:
            print("No val_loss_0 measurements found.")
    else:
        if target_epoch == "latest":
            if latest_val_loss is not None:
                print(f"Latest val_loss_0: {latest_val_loss:.6f} at epoch {latest_epoch}")
            else:
                print("No val_loss_0 measurements found.")
        elif val_loss_at_epoch is not None:
            print(f"val_loss_0 at epoch {target_epoch}: {val_loss_at_epoch:.6f}")
        else:
            print(f"No val_loss_0 recorded for epoch {target_epoch}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find val_loss_0 in wandb logs.")
    parser.add_argument("wandb_run_dir", type=str, help="Path to the wandb run directory")
    parser.add_argument("--epoch", type=str, default=None, help="Epoch to retrieve val_loss_0 for (or 'latest')")
    args = parser.parse_args()
    
    find_val_loss(args.wandb_run_dir, args.epoch)
