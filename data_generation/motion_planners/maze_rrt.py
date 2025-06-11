import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


from data_generation.motion_planners.base_rrt import BaseRRT
from data_generation.maze.maze_environment import MazeEnvironment

class MazeRRT(BaseRRT):
    def __init__(self, 
                 maze: MazeEnvironment,
                 source: np.ndarray, 
                 max_step_size: float=0.1):
        super().__init__(source, max_step_size)
        self.maze = maze

    def sample_free(self) -> np.ndarray:
        """
        Returns a random point in the free space.
        """
        return self.maze.sample_collision_free_point()

    def is_free(self, q):
        """
        Returns True if the configuration q is in the free space.
        """
        return not self.maze.in_collision(q, mode='padded_obstacles')

    def visualize(self, path=None):
        fig, ax = plt.subplots()

        # plot the maze environment
        polygons_to_plot = self.maze.obstacles_vpolytopes
        background_color = 'white'
        polygons_color = 'black'
        
        ax.set_facecolor(background_color)
        for polygon in polygons_to_plot:
            v = polygon.vertices().transpose()
            hull = ConvexHull(v)
            plt.fill(*(v[hull.vertices].transpose()), 
                     facecolor=polygons_color)
        plt.xlim(self.maze.bounds[0])
        plt.ylim(self.maze.bounds[1])
        ax.set_aspect('equal', adjustable='box')

        # plot the RRT tree
        for vertex in self.vertices:
            if vertex.parent is not None:
                plt.plot([vertex.value[0], vertex.parent.value[0]], 
                         [vertex.value[1], vertex.parent.value[1]], 
                         'ro-', markersize=1)
                
        # plot the path
        if path is not None:
            for i in range(len(path) - 1):
                plt.plot([path[i][0], path[i + 1][0]], 
                         [path[i][1], path[i + 1][1]], 
                         'bo-', markersize=1)
        
        plt.show()

def main():
    import data_generation.maze.gcs_utils as gcs_utils
    obstacles = gcs_utils.create_test_box_env()
    bounds = np.array([[0, 5], [0, 5]])
    maze_env = MazeEnvironment(bounds, obstacles=obstacles, 
                            obstacle_padding=0.1)
    source = np.array([2.5, 2.5])
    maze_rrt = MazeRRT(maze_env, source)

    # grow tree to goal
    path = maze_rrt.grow_to_goal(np.array([4.5, 4.5]))
    maze_rrt.visualize(path=path)
    
    # test reset function
    maze_rrt.reset()
    path = maze_rrt.grow_to_goal(np.array([2.5, 2.7]))
    maze_rrt.visualize(path=path)
    
    # add more nodes
    maze_rrt.reset()
    maze_rrt.grow(N=1000)
    q_goal = maze_env.sample_collision_free_point()
    path = maze_rrt.find_path(q_goal, num_shortcut_attempts=0)
    maze_rrt.visualize(path=path)

    # test shortcutting
    shortcut_path = maze_rrt.shortcut_path(path)
    print(f'Original path length: {len(path)}')
    print(f'Shortcut path length: {len(shortcut_path)}')
    maze_rrt.visualize(path=shortcut_path)



if __name__ == '__main__':
    main()