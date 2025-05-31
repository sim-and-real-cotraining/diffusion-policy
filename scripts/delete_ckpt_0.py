import os
import argparse

"""
Usage:
------
This script scans a nested directory structure to find "wandb" run directories
that contain a "checkpoints" subdirectory. It then deletes any checkpoint files 
inside the "checkpoints" directory that start with "epoch=0000".

Example usage:
--------------

    python delete_ckpt_0.py /path/to/wandb/root

This will:
1. Locate all "wandb" run directories that contain a "checkpoints" folder.
2. Delete any checkpoint files in "checkpoints" that start with "epoch=0000".
3. Print the deleted files for confirmation.

Arguments:
----------
- root_dir (str): The path to the root directory containing wandb runs.
"""

def find_wandb_leaf_dirs(root_dir):
    """
    Recursively find leaf directories that contain a 'wandb' run.
    A leaf directory is one that has a 'checkpoints' subdirectory.
    """
    leaf_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "checkpoints" in dirnames:  # Ensure it's a leaf by checking for 'checkpoints' folder
            leaf_dirs.append(os.path.join(dirpath, "checkpoints"))
    return leaf_dirs

def delete_ckpt_files(leaf_dirs):
    """
    Delete all checkpoint files in each leaf directory that start with 'epoch=0000'.
    """
    for checkpoint_dir in leaf_dirs:
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.startswith("epoch=0000") and file.endswith(".ckpt"):
                    file_path = os.path.join(checkpoint_dir, file)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Remove 'epoch=0000' checkpoint files from wandb runs.")
    parser.add_argument("root_dir", type=str, help="Path to the root directory containing wandb runs.")
    args = parser.parse_args()

    print(f"Scanning for wandb runs in: {args.root_dir}")
    leaf_dirs = find_wandb_leaf_dirs(args.root_dir)
    print(f"Found {len(leaf_dirs)} wandb runs with checkpoint directories.")
    
    delete_ckpt_files(leaf_dirs)
    print("Cleanup completed.")

if __name__ == "__main__":
    main()
