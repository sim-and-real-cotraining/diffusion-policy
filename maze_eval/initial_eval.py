"""
Usage:
python maze_eval/initial_eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import zarr
import wandb
import json
import tqdm
import pickle
import shutil

from collections import deque
from torch.utils.data import DataLoader
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from data_generation.maze.gcs_utils import *

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
# @click.option('-d', '--device', default='cuda:0')
@click.option('-d', '--device', default='cpu')
def main(checkpoint, output_dir, device):
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    # set up output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(f'{os.path.dirname(checkpoint)}/../.hydra/config.yaml', 
                    os.path.join(output_dir, 'config.yaml'))
    pathlib.Path(os.path.join(output_dir, 'plots')).mkdir()

    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get normalizer
    dataset: BaseLowdimDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseLowdimDataset)
    normalizer = dataset.get_normalizer()

    # get policy from workspace
    policy = workspace.model
    policy.set_normalizer(normalizer)
    if cfg.training.use_ema:
        policy = workspace.ema_model
        policy.set_normalizer(normalizer)
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # run eval
    # get initial starting and end positions
    # test_traj = zarr.open(cfg.task.dataset.zarr_path, mode='r')
    # source = test_traj['data']['state'][0]
    # target = test_traj['data']['state'][-1]

    # get environment
    regions = create_env()
    bounds = np.array([[0, 5], [0, 5]])

    # get parameters
    horizon = cfg.policy.horizon
    action_dim = cfg.policy.action_dim
    obs_dim = cfg.policy.obs_dim
    action_horizon = cfg.policy.n_action_steps
    obs_horizon = cfg.policy.n_obs_steps
    use_target_cond = cfg.policy.use_target_cond
    D_t = cfg.policy.target_dim
    B = 1 # batch size is 1
    num_traj = 100

    

    with torch.no_grad():
        all_trajectories = []
        passed_all_tests = []
        passed_collision_tests = []
        passed_velocity_tests = []
        failed_to_converge = []

        eps = 0.1
        for i in tqdm.tqdm(range(num_traj)):
            done = False
            dt = 0.1

            source = sample_collision_free_point(regions, bounds)
            target = None
            if use_target_cond:
                target = sample_collision_free_point(regions, bounds)
            else:
                target = np.array([2.0, 1.0])
            distance = np.linalg.norm(source - target)

            waypoints = [source]
            obs_deque = deque([torch.from_numpy(source).reshape(B,1,2)] * obs_horizon,
                            maxlen=obs_horizon)
            while len(waypoints) <= 300:
                obs_dict = None
                if use_target_cond:
                    obs_dict = deque_to_dict(obs_deque, target)
                else:
                    obs_dict = deque_to_dict(obs_deque)
                action_seq = policy.predict_action(obs_dict)['action_pred'][0]
                for action in action_seq:
                    waypoints.append(action.cpu().detach().numpy())
                    obs_deque.append(action.reshape(B,1,2))

                    dist_to_target = np.linalg.norm(action.cpu().detach().numpy() - target)
                    vel_estimate = np.linalg.norm((waypoints[-1]-waypoints[-2])/dt)
                    done = dist_to_target < eps and vel_estimate < eps
                    if done:
                        break

                if done:
                    break

            waypoints = np.array(waypoints).transpose()

            # check passed conditions
            collidion_indices = get_colliding_indices(regions, waypoints)
            collision_free = len(collidion_indices) == 0
            satisfies_velocity_bounds = check_velocity_bounds(
                waypoints, velocity_bounds=np.array([[-1,1],[-1,1]]), dt=dt
            )
            passed_all = collision_free and satisfies_velocity_bounds and done

            if done == False:
                failed_to_converge.append(i)
            if collision_free:
                passed_collision_tests.append(i)
            if satisfies_velocity_bounds:
                passed_velocity_tests.append(i)
            if passed_all:
                passed_all_tests.append(i)

            # save plots and add trajectory
            save_trajectory_plot(f"{output_dir}/plots/{i:0>3}.png", 
                                 regions, bounds, source, target, waypoints,
                                 velocity_bounds=np.array([[-1,1],[-1,1]]),
                                 collision_indices=collidion_indices,
                                 dt=0.1)
            all_trajectories.append((source, target, waypoints))
        
        # Compute statistics and write to logs
        log_eval_results(
            f"{output_dir}/eval_results.txt",
            passed_all_tests,
            passed_collision_tests,
            passed_velocity_tests,
            failed_to_converge,
            num_traj,
        )

        # pickle and save trajectories
        trajectories_path = os.path.join(output_dir, 'eval_trajectories.pkl')
        with open(trajectories_path, 'wb') as f:
            pickle.dump(all_trajectories, f)

def deque_to_dict(obs_deque: deque, target=None) -> dict[str, torch.Tensor]:
    obs_array = torch.cat(list(obs_deque), axis=1)
    if target is None:
        return {'obs': obs_array}
    else:
        target = torch.from_numpy(target).reshape(1,-1)
        return {'obs': obs_array, 'target': target}
    
if __name__ == '__main__':
    main()
