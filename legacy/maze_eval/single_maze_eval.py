"""
Usage:
python single_maze_eval/single_maze_eval.py --checkpoint <checkpoint> \
    -o <output_dir>
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
import tqdm
import pickle
import shutil
import time

from collections import deque
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from data_generation.maze.gcs_utils import *
from data_generation.maze.maze_environment import MazeEnvironment

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
# @click.option('-d', '--device', default='cpu')
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
    workspace: BaseWorkspace
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get normalizer: this might be expensive for larger datasets
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

    # get parameters
    obs_horizon = cfg.policy.n_obs_steps
    B = 1 # batch size is 1
    num_traj = 100

    with torch.no_grad():
        all_trajectories = []
        passed_all_tests = []
        passed_collision_tests = []
        passed_velocity_tests = []
        failed_to_converge = []

        eps = 0.1
        dt = 0.1

        obstacles = create_test_box_env()
        bounds = np.array([[0.0, 5.0], [0.0, 5.0]])
        maze = MazeEnvironment(bounds, obstacles, obstacle_padding=0.1)
        maze_no_padding = MazeEnvironment(bounds, obstacles, obstacle_padding=0.0)
        source = np.array([2.5, 2.5]) # TODO: read this from data generatino config
        regions = maze_no_padding.regions

        for i in tqdm.tqdm(range(num_traj)):
            done = False

            target = maze.sample_end_point()
            waypoints = [source]

            # Create observation deques
            state_deque = deque([torch.from_numpy(source).reshape(B,1,2)] * obs_horizon,
                            maxlen=obs_horizon)
            
            while len(waypoints) <= 300:
                obs_dict = deque_to_dict(state_deque, target)
                action_seq = policy.predict_action(obs_dict)['action_pred'][0]
                for action in action_seq:
                    waypoints.append(action.cpu().detach().numpy())

                    # update deques
                    state_deque.append(action.reshape(B,1,2))

                    # check success conditions (arrived near target with low velocity)
                    dist_to_target = np.linalg.norm(action.cpu().detach().numpy() - target)
                    vel_estimate = np.linalg.norm((waypoints[-1]-waypoints[-2])/dt)
                    done = dist_to_target < eps and vel_estimate < eps
                    if done:
                        break

                if done:
                    break

            # check passed conditions
            waypoints = np.array(waypoints).transpose()
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
            # gcs_regions = maze.regions
            gcs_regions = None
            save_trajectory_plot(f"{output_dir}/plots/{i:0>3}.png", 
                                 regions, bounds, source, target, waypoints,
                                 velocity_bounds=np.array([[-1,1],[-1,1]]),
                                 collision_indices=collidion_indices,
                                 dt=0.1,
                                 gcs_regions=gcs_regions)
            all_trajectories.append((source, target, waypoints, maze))
        
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
