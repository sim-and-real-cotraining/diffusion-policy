import os
import json
import argparse
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

"""
Example usage
python transfer_checkpoints.py --remote awei@txe1-login.mit.edu \
    --src_path /home/gridsan/awei/workspace/gcs-diffusion/data/outputs/cotrain/50_500_1_3 \
        --dst_path data/outputs/cotrain/test_move_checkpoint/

By default, the script will transfer the best val_loss_0 and val_ddim_mse_0 checkpoints
"""

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Script to transfer files from a remote machine and process them.')
    
    # Add arguments
    parser.add_argument('--remote', type=str, required=True, 
                        help='Remote machine and user (e.g., user@txe1-login.mit.edu)')
    parser.add_argument('--src_path', type=str, required=True, 
                        help='Source path on the remote machine')
    parser.add_argument('--dst_path', type=str, required=True, 
                        help='Destination path on the local machine')
    parser.add_argument('--keys', type=str, nargs='+', default=['val_loss_0', 'val_ddim_mse_0'], 
                        help='List of keys to find best checkpoints (e.g., val_loss_0 val_ddim_mse_0)')
    parser.add_argument('--top_ks', type=int, nargs='+', default=None, 
                        help='Top K values to track for each key (default: 1 for all keys)')
    parser.add_argument('--modes', type=str, nargs='+', default=None, 
                        help='Modes for each key (default: min for all keys')
    parser.add_argument('--transfer_latest', type=bool, default=True, 
                        help='Whether to transfer the latest ckpt (default: True)')
    parser.add_argument('--plot_only', action='store_true', 
                        help='Only plot the training curves (default: False)')
    args = parser.parse_args()

    remote = args.remote
    src_path = args.src_path
    dst_path = args.dst_path
    monitor_keys = args.keys
    top_ks = [1] * len(monitor_keys) if args.top_ks is None else args.top_ks
    modes = ['min'] * len(monitor_keys) if args.modes is None else args.modes
    transfer_latest = args.transfer_latest
    plot_only = args.plot_only
    
    # Validate arguments
    assert len(monitor_keys) == len(top_ks) == len(modes), "Lengths of keys, top_ks, and modes must be the same"
    for mode in modes:
        assert mode in ['min', 'max'], f"Mode {mode} not supported"
    for top_k in top_ks:
        assert top_k > 0, "Top K must be greater than 0"

    if not os.path.exists(dst_path):
        # Create directories
        os.makedirs(f'{dst_path}/checkpoints')
    else:
        # Ask user if they want to remove directory
        print(f"Directory {dst_path} already exists. Would you like to remove it?")
        response = input("y/n: ")
        if response == 'y':
            os.system(f'rm -rf {dst_path}')
            os.makedirs(f'{dst_path}/checkpoints')
        else:
            print("Exiting...")
            return
    
    # Transfer normalizer
    os.system(f'scp {remote}:{src_path}/normalizer.pt {dst_path}')

    # Transfer hydra config
    os.system(f'scp -r {remote}:{src_path}/.hydra {dst_path}')
    cfg = OmegaConf.load(f'{dst_path}/.hydra/config.yaml')

    # Identify best checkpoints (amongst saved checkpoints)
    best_checkpoints = {key: [] for key in monitor_keys}
    os.system(f'scp {remote}:{src_path}/logs.json.txt {dst_path}')

    with open(f'{dst_path}/logs.json.txt', 'r') as f:
        lines = f.readlines()
    
    epoch_axis = {key: [] for key in monitor_keys}
    value_axis = {key: [] for key in monitor_keys}
    for line in lines:
        for i, key in enumerate(monitor_keys):
            if key in line:
                # Extract values and epoch
                epoch, value = get_epoch_and_value(line, key)
                epoch_axis[key].append(epoch)
                value_axis[key].append(value)
                if epoch % cfg.training.checkpoint_every == 0:
                    best_checkpoints[key].append((epoch, value))
    
    # Sort the checkpoints
    for key in monitor_keys:
        if modes[i] == 'min':
            best_checkpoints[key].sort(key=lambda x: x[1])
        elif modes[i] == 'max':
            best_checkpoints[key].sort(key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f"Mode {modes[i]} not supported")

        checkpoints = [best_checkpoints[key][i] for i in range(top_ks[i])]
        best_checkpoints[key] = checkpoints
    
    # Plot the training curves
    fig, axs = plt.subplots(len(monitor_keys))
    for i, key in enumerate(monitor_keys):
        axs[i].plot(epoch_axis[key], value_axis[key])
        axs[i].set_title(key)
    
    # highlight the best and saved checkpoints on the plot
    for key_idx, key in enumerate(monitor_keys):
        for top_k_idx in range(top_ks[key_idx]):
            epoch, value = best_checkpoints[key][top_k_idx]
            axs[key_idx].scatter(epoch, value, c='r', marker='*', zorder=100)
            axs[key_idx].annotate(f'{top_k_idx+1}', (epoch, value), textcoords="offset points", xytext=(0,10), ha='center')
        for i in range(len(epoch_axis[key])):
            epoch = epoch_axis[key][i]
            value = value_axis[key][i]
            if epoch % cfg.training.checkpoint_every == 0 and epoch not in [x[0] for x in best_checkpoints[key]]:
                axs[key_idx].scatter(epoch, value, color='g')
    
    plt.savefig(f'{dst_path}/training_plot.png')

    # Transfer checkpoints
    if not plot_only:
        epoches_to_save = set()
        for key_idx, key in enumerate(monitor_keys):
            for i in range(top_ks[key_idx]):
                epoches_to_save.add(best_checkpoints[key][i][0])

        for epoch in epoches_to_save:
            os.system(f'scp {remote}:{src_path}/checkpoints/epoch={epoch:04d}* {dst_path}/checkpoints')
        if transfer_latest:
            os.system(f'scp {remote}:{src_path}/checkpoints/latest.ckpt {dst_path}/checkpoints')

    # Show the training curves
    plt.show()

def get_epoch_and_value(line, key):
    """
    Get monitor value from line
    """
    log_dict = json.loads(line)
    return log_dict['epoch'], log_dict[key]
    

if __name__ == '__main__':
    """
    Run script from local computer
    """
    main()