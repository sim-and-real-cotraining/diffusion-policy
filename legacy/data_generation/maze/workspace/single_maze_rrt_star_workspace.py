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
from data_generation.motion_planners.maze_rrt_star import MazeRRTStar

class SingleMazeRRTStarWorkspace:
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

        # RRT Star settings
        self.reset_every_trajectory = cfg.reset_every_trajectory
        self.max_samples = cfg.max_samples
        self.num_shortcut_attempts = cfg.num_shortcut_attempts

        self.tqdm_interval = cfg.tqdm_interval

    def run(self):
        # Generate the random maze environments

        """
        Generates data and saves as zarr

        <data_dir>
        ├── single_maze_rrt_star.zarr
            ├── data
                ├── state
                ├── action
                |── target
            ├── meta
                ├── episode_ends
        ├── config.yaml
        """
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

        maze_rrt_star = MazeRRTStar(self.maze, self.source)
        if not self.reset_every_trajectory:
            maze_rrt_star.grow(N=self.max_samples)

        data = []
        path = None
        pbar = tqdm(total=self.num_trajectories, 
                position=0,
                miniters=self.tqdm_interval,
                desc='Trajectory generation')
                
        while len(data) < self.num_trajectories:
            goal = self.maze.sample_collision_free_point()
            if self.reset_every_trajectory:
                maze_rrt_star.reset()
                path = maze_rrt_star.grow_to_goal(goal,
                            max_samples=self.max_samples,
                            num_shortcut_attempts=self.num_shortcut_attempts)
            else:
                path = maze_rrt_star.find_path(goal, 
                            num_shortcut_attempts=self.num_shortcut_attempts)
            
            if path is not None:
                waypoints = self._rrt_path_to_array(path)
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

    def _rrt_path_to_array(self, path):
        # uses same velocity box constraints as gcs: v \in [-1, 1]
        # assumes dt = 0.1 => max step size in any direction is 0.1
        assert path is not None and len(path) >= 2
        waypoints = []
        for i in range(len(path)-1):
            delta = path[i+1] - path[i]
            max_direction = np.argmax(np.abs(delta))
            max_delta = np.abs(delta[max_direction])
            num_steps = max(1, int(max_delta // 0.1))
            for step in range(num_steps):
                waypoints.append(path[i] + delta * (step / num_steps))
        waypoints.append(path[-1])
        return np.array(waypoints)


def main():
    # visualize some trajectories
    # this should be run after collecting data with generate_maze_data.py   
    obstacles = gcs_utils.create_test_box_env()
    bounds = np.array([[0, 5], [0, 5]])
    maze = MazeEnvironment(bounds, obstacles=obstacles,
                            obstacle_padding=0.0)
    
    # read from disk
    dataset = zarr.open(
        'data_generation/maze_data_rrt_star/rrt_star.zarr', 
        mode='r')
    current_start = 0
    for i in range(dataset['meta/episode_ends'].shape[0]):
        current_end = dataset['meta/episode_ends'][i]
        trajectory = dataset['data/state'][current_start:current_end]
        source = trajectory[0]
        target = dataset['data/target'][current_start]
        current_start = current_end

        maze.plot_trajectory(start=source,
                             end=target,
                             waypoints=trajectory,
                             mode='obstacles')
        # maze.animate_trajectory(trajectory, source, target)
        
if __name__ == '__main__':
    main()