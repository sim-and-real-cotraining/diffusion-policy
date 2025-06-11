import numpy as np
import time
import os
import zarr
from datetime import datetime
import logging

from tqdm import tqdm
from omegaconf import OmegaConf

import data_generation.maze.gcs_utils as gcs_utils
from data_generation.maze.maze_environment import MazeEnvironment
from pydrake.geometry.optimization import (Point,
                                           GraphOfConvexSetsOptions)
from pydrake.planning import GcsTrajectoryOptimization

class SingleMazeGCSWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg

        obstacles = gcs_utils.create_test_box_env()
        bounds = np.array([[0, 5], [0, 5]])
        self.maze = MazeEnvironment(bounds, obstacles=obstacles, 
                                obstacle_padding=0.1)

        self.source = np.array(cfg.source)
        self.num_trajectories = cfg.num_trajectories

        if cfg.task_id is not None:
            cfg.data_dir = f'{cfg.data_dir}/{cfg.task_id}'
        self.data_dir = cfg.data_dir
        self.zarr_name = cfg.zarr_name

        # GCS settings
        self.max_rounded_paths = cfg.max_rounded_paths
        self.max_velocity = cfg.max_velocity
        self.continuity_order = cfg.continuity_order
        self.bezier_order = cfg.bezier_order

        self.tqdm_interval = cfg.tqdm_interval

    def run(self):
        # Generate the random maze environments

        """
        Generates data and saves as zarr

        <data_dir>
        ├── single_maze_gcs.zarr
            ├── data
                ├── state
                ├── action
                |── target
            ├── meta
                ├── episode_ends
        ├── config.yaml
        """
        logging.getLogger('drake').setLevel(logging.WARNING)
        data = self.generate_data()
        
        # Save data as zarr and config file
        print("Saving data and config file")
        self.save_data_as_zarr(data)
        self.save_config()
        print("All done")
    
    def generate_data(self):
        """
        Generate data for a single maze and single source

        Returns an array of tuples (target, trajectory)
        """
        seed = int((datetime.now().timestamp() % 1) * 1e6)
        np.random.seed(seed)

        data = []
        pbar = tqdm(total=self.num_trajectories, 
                position=0,
                miniters=self.tqdm_interval,
                desc='Trajectory generation')
        
        # Build graph
        start_time = time.time()
        gcs = GcsTrajectoryOptimization(self.maze.dim)
        free_space = gcs.AddRegions(self.maze.regions, self.bezier_order)
        source_region = gcs.AddRegions([Point(self.source)], 0)
        gcs.AddEdges(source_region, free_space)

        # Cost & Constraints
        gcs.AddVelocityBounds(np.array([-self.max_velocity, -self.max_velocity]),
                            np.array([self.max_velocity, self.max_velocity]))
        gcs.AddPathContinuityConstraints(self.continuity_order)
        gcs.AddTimeCost()
        gcs.AddPathLengthCost()
        time_to_build_graph = time.time() - start_time

        # Configure Options
        options = GraphOfConvexSetsOptions()
        options.max_rounded_paths = self.max_rounded_paths
        
        solves_since_rebuild = 0
        solves_per_build = 7 # computed roughly as sqrt(2*time_to_build_graph / rate_of_solve_time_increase)
        while len(data) < self.num_trajectories:
            goal = self.maze.sample_end_point()
            target_region = gcs.AddRegions([Point(goal)], 0)
            gcs.AddEdges(free_space, target_region)
            [traj, result] = gcs.SolvePath(source_region, target_region, options)
            solves_since_rebuild += 1

            # Rebuild GCS
            if solves_since_rebuild == 7:
                gcs = GcsTrajectoryOptimization(self.maze.dim)
                free_space = gcs.AddRegions(self.maze.regions, self.bezier_order)
                source_region = gcs.AddRegions([Point(self.source)], 0)
                gcs.AddEdges(source_region, free_space)
                gcs.AddVelocityBounds(np.array([-self.max_velocity, -self.max_velocity]),
                                    np.array([self.max_velocity, self.max_velocity]))
                gcs.AddPathContinuityConstraints(self.continuity_order)
                gcs.AddTimeCost()
                gcs.AddPathLengthCost()
                solves_since_rebuild = 0

            
            if not result.is_success():
                continue

            # Successful trajectory => collect the data
            waypoints = gcs_utils.composite_trajectory_to_array(traj).transpose()
            data.append((goal, waypoints))
            pbar.update(1)
        
        pbar.close()
        return data
    
    def save_data_as_zarr(self, data):
        # compute episode ends
        current_end = 0
        episode_ends = []
        for _, trajectory in data:
            current_end += trajectory.shape[0]
            episode_ends.append(current_end)

        # compute targets
        target = np.zeros((episode_ends[-1], self.maze.dim))
        current_start = 0
        for i, end in enumerate(episode_ends):
            target[current_start:end, :] = data[i][0]
            current_start = end
        
        # compute states and actions
        state = np.concatenate([trajectory for _, trajectory in data])
        assert state.shape[0] == episode_ends[-1] \
            and state.shape[1] == self.maze.dim
        
        action = np.zeros((episode_ends[-1], self.maze.dim))
        start_idx = 0
        for end_idx in episode_ends:
            trajectory = state[start_idx:end_idx, :]
            shifted_trajectory = \
                np.concatenate([trajectory[1:, :], trajectory[-1:, :]], axis=0)
            action[start_idx:end_idx, :] = shifted_trajectory
            start_idx = end_idx
        
        # save to zarr
        zarr_path = os.path.join(self.data_dir, self.zarr_name)
        root = zarr.open(zarr_path, mode='w')
        data_group = root.create_group('data')
        meta_group = root.create_group('meta')

        state_chunk_size = (1024, 2)
        action_chunk_size = (2048, 2)
        target_chunk_size = (1024, 2)
        data_group.create_dataset('state', data=state, chunks=state_chunk_size)
        data_group.create_dataset('action', data=action, chunks = action_chunk_size)
        data_group.create_dataset('target', data=target, chunks = target_chunk_size)
        meta_group.create_dataset('episode_ends', data=episode_ends)
    
    def save_config(self):
        config_path = os.path.join(self.data_dir, 'config.yaml')
        OmegaConf.save(self.cfg, config_path)

def main():
    # visualize some trajectories
    # this should be run after collecting data with generate_maze_data.py
    import random
    obstacles = gcs_utils.create_test_box_env()
    bounds = np.array([[0, 5], [0, 5]])
    maze = MazeEnvironment(bounds, obstacles=obstacles,
                            obstacle_padding=0.0)
    
    # read from disk
    dataset = zarr.open(
        'data/single_maze/gcs.zarr', 
        mode='r')
    current_start = None
    while True:
        i = random.randint(0, len(dataset['meta/episode_ends'])-1)
        current_start = 0 if i == 0 else dataset['meta/episode_ends'][i-1]
        current_end = dataset['meta/episode_ends'][i]
        trajectory = dataset['data/state'][current_start:current_end]
        source = trajectory[0]
        target = dataset['data/target'][current_start]
        current_start = current_end

        maze.plot_trajectory(start=source,
                             end=target,
                             waypoints=trajectory,
                             mode='obstacles')
        
if __name__ == '__main__':
    main()