import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

from typing import (List, Dict, Union)
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from datetime import datetime

from pydrake.geometry.optimization import (HPolyhedron,
                                           VPolytope,
                                           Point)

Polyhedrons = List[HPolyhedron]

class MazeEnvironment:
    """
    Class for 2D maze environments with axis aligned rectangular obstacles.
    
    Assumptions:
    - Obstacles are axis aligned rectangles
    - The vertices of the obstacles are rounded to the nearest 10th
    """

    def __init__(self, bounds: np.ndarray, 
                 obstacles: Polyhedrons=None, 
                 regions: Polyhedrons=None,
                 obstacle_padding: float=0.0) -> None:
        assert obstacles is not None or regions is not None
        assert bounds.shape[0] == 2 # only support 2 dimensional mazes

        self.dim = 2
        self.bounds = bounds
        self.padding = obstacle_padding
        
        self.obstacles = obstacles
        self.obstacles_vpolytopes = None
        self.padded_obstacles = None
        self.padded_obstacles_vpolytopes = None
        if obstacles is not None:
            self.obstacles_vpolytopes = self.convert_to_vpolytope(obstacles)
            self.padded_obstacles = self.get_padded_obstacles(
                self.obstacles_vpolytopes, obstacle_padding, bounds)
            self.padded_obstacles_vpolytopes = self.convert_to_vpolytope(
                self.padded_obstacles)
        
        self.regions = regions
        if regions is None:
            assert obstacles is not None
            self.regions = self.get_regions_from_obstacles()
        self.regions_vpolytopes = self.convert_to_vpolytope(self.regions)

        if regions is None:
            self.binary_maze = self.construct_binary_maze_representation()
        # print_binary_img(self.binary_maze)

        # Variables for image cropping (see to_img function)
        # This is a hacky solution...
        self.left_crop, self.right_crop = None, None
        self.top_crop, self.bot_crop = None, None


    """ Constructor functions """
    def get_regions_from_obstacles(self, try_both = True) -> Polyhedrons:
        """ Generates a convex decomposition of the environment from the padded
        obstacles.
        
        This is a naive implementation that does not generate a minimal cover.
        The algorithm is run in both the vertical and the horizontal direction.
        The decomposition with the fewest number of regions will be used.

        try_both: determines whether to try to generate regions horizontally and vertically
        if true: generate regions horizontally and vertically. Pick the oritentation with
        less regions
        if false: only generate regions vertically
        """

        assert self.padded_obstacles is not None
        
        # extract and sort all the splits
        vertical_planes = [*self.bounds[0]]
        horizontal_planes = [*self.bounds[1]]

        for v_obstacle in self.padded_obstacles_vpolytopes:
            planes = self.get_planes(v_obstacle)
            vertical_planes.append(planes['left'])
            vertical_planes.append(planes['right'])
            horizontal_planes.append(planes['bot'])
            horizontal_planes.append(planes['top'])

        vertical_planes = sorted(set(vertical_planes))
        horizontal_planes = sorted(set(horizontal_planes))

        # Try vertical version of algorithm
        # initial region creation
        initial_regions = {}
        # search each vertical strip
        for i in range(len(vertical_planes)-1):
            # search each horizontal strip
            planes = {'left': vertical_planes[i], 
                      'right': vertical_planes[i+1]}
            for j in range(len(horizontal_planes)-1):
                # check if block is in obstacle
                center = np.array([
                    (vertical_planes[i]+vertical_planes[i+1])/2,
                    (horizontal_planes[j]+horizontal_planes[j+1])/2
                ])
                is_free_space = not self.in_collision(center, mode='padded_obstacles')
                if is_free_space and 'bot' not in planes:
                    # start of new region
                    planes['bot'] = horizontal_planes[j]
                elif not is_free_space and 'bot' in planes:
                    # end of vertical region, add it to the dictionary
                    planes['top'] = horizontal_planes[j]
                    height = planes['top'] - planes['bot']
                    key = (planes['bot'], height)
                    item = copy.deepcopy(planes)
                    if key in initial_regions:
                        initial_regions[key].append(item)
                    else:
                        initial_regions[key] = [item]
                                        
                    # reset planes
                    del planes['bot']
                    del planes['top']
            
            # Check if plane reaches top of environment
            if 'bot' in planes:
                planes['top'] = horizontal_planes[-1]
                height = planes['top'] - planes['bot']
                key = (planes['bot'], height)
                item = copy.deepcopy(planes)
                if key in initial_regions:
                    initial_regions[key].append(item)
                else:
                    initial_regions[key] = [item]
        
        # merge regions horizontally
        planes_vertical_direction = []
        for key, planes_list in initial_regions.items():
            # note that planes_list is sorted left to right
            # since elements were added left to right
            planes = {
                'bot': key[0], # bot
                'top': key[0]+key[1], # bot + height
                'left': planes_list[0]['left'],
                'right': planes_list[0]['right']
            }
            for i in range(1, len(planes_list)):
                # check for intersection
                if planes_list[i]['left'] <= planes['right']:
                    # merge regions
                    planes['right'] = planes_list[i]['right']
                else:
                    # no intersection => finished merging this region
                    planes_vertical_direction.append(copy.deepcopy(planes))
                    # reset planes
                    planes['left'] = planes_list[i]['left']
                    planes['right'] = planes_list[i]['right']
            
            # add the last region
            planes_vertical_direction.append(copy.deepcopy(planes))
        
        if not try_both:
            return [self.planes_to_hpolyhedron(planes) for planes in planes_vertical_direction]

        # Try horizontal version of algorithm
        # initial region creation
        initial_regions = {}
        # search each horizontal strip
        for i in range(len(horizontal_planes)-1):
            # search each vertical strip
            planes = {'bot': horizontal_planes[i], 
                      'top': horizontal_planes[i+1]}
            for j in range(len(vertical_planes)-1):
                # check if block is in obstacle
                center = np.array([
                    (vertical_planes[j]+vertical_planes[j+1])/2,
                    (horizontal_planes[i]+horizontal_planes[i+1])/2
                ])
                is_free_space = not self.in_collision(center, mode='padded_obstacles')
                if is_free_space and 'left' not in planes:
                    # start of new region
                    planes['left'] = vertical_planes[j]
                elif not is_free_space and 'left' in planes:
                    # end of horizontal region, add it to the dictionary
                    planes['right'] = vertical_planes[j]
                    width = planes['right'] - planes['left']
                    key = (planes['left'], width)
                    item = copy.deepcopy(planes)
                    if key in initial_regions:
                        initial_regions[key].append(item)
                    else:
                        initial_regions[key] = [item]
                                        
                    # reset planes
                    del planes['left']
                    del planes['right']
            
            # Check if plane reaches top of environment
            if 'left' in planes:
                planes['right'] = horizontal_planes[-1]
                width = planes['right'] - planes['left']
                key = (planes['left'], width)
                item = copy.deepcopy(planes)
                if key in initial_regions:
                    initial_regions[key].append(item)
                else:
                    initial_regions[key] = [item]
        
        # merge regions vertically
        planes_horizontal_direction = []
        for key, planes_list in initial_regions.items():
            # note that planes_list is sorted left to right
            # since elements were added left to right
            planes = {
                'left': key[0], # left
                'right': key[0]+key[1], # left + width
                'bot': planes_list[0]['bot'],
                'top': planes_list[0]['top']
            }
            for i in range(1, len(planes_list)):
                # check for intersection
                if planes_list[i]['bot'] <= planes['top']:
                    # merge regions
                    planes['top'] = planes_list[i]['top']
                else:
                    # no intersection => finished merging this region
                    planes_horizontal_direction.append(copy.deepcopy(planes))
                    # reset planes
                    planes['bot'] = planes_list[i]['bot']
                    planes['top'] = planes_list[i]['top']
            
            # add the last region
            planes_horizontal_direction.append(copy.deepcopy(planes))
        
        # return the decomposition with less regions
        planes_to_convert = []
        if len(planes_vertical_direction) < len(planes_horizontal_direction):
            planes_to_convert = planes_vertical_direction
        else:
            planes_to_convert = planes_horizontal_direction
        
        return [self.planes_to_hpolyhedron(planes) for planes in planes_to_convert]

    def construct_binary_maze_representation(self):
        W = int(10*self.bounds[0][1]+2) # +2 for border
        H = int(10*self.bounds[1][1]+2) # +2 for border
        binary_maze = np.ones((H, W), dtype='int8')
        # loop through interior points
        for i in range(1,H-1):
            for j in range(1,W-1):
                point = np.array([j/10 - 1/20, i/10 - 1/20])
                if not self.in_collision(point, mode='obstacles'):
                    binary_maze[H-i-1,j] = 0
        return binary_maze.astype('int8')
    
    def get_binary_maze_representation(self, position: np.ndarray=None) -> np.ndarray:
        if position is None:
            return self.binary_maze
        
        assert self.bounds[0][0] <= position[0] <= self.bounds[0][1]
        assert self.bounds[1][0] <= position[1] <= self.bounds[1][1]
        assert self.bounds[0][1] == self.bounds[1][1] # assume picture is square

        img_H = self.binary_maze.shape[0]
        robot_pixel_pos = np.minimum(10*position+1, img_H-2).astype(int)
        binary_img = np.copy(self.binary_maze)
        binary_img[img_H-1-robot_pixel_pos[1], robot_pixel_pos[0]] = 2
        return binary_img
                
    """ Sampling functions """
    def sample_start_point(self, bounds: np.array=None) -> np.ndarray:
        return self.sample_collision_free_point(bounds)

    def sample_end_point(self, bounds: np.array=None) -> np.ndarray:
        return self.sample_collision_free_point(bounds)

    def sample_collision_free_point(self, bounds: np.array=None) -> np.ndarray:
        if bounds is None:
            bounds = self.bounds
        while True:
            sample = np.random.uniform(bounds[:,0], bounds[:,1])
            if not self.in_collision(sample, mode='regions'):
                return sample
            
        
    """ Plotting functions """
    def plot_environment(self, mode: str) -> None:
        fig, ax = plt.subplots()
        if mode == 'regions':
            polygons_to_plot = self.regions_vpolytopes
            background_color = 'black'
            polygons_color = 'white'
        elif mode == 'obstacles':
            polygons_to_plot = self.obstacles_vpolytopes
            background_color = 'white'
            polygons_color = 'black'
        elif mode == 'padded_obstacles':
            polygons_to_plot = self.padded_obstacles_vpolytopes
            background_color = 'white'
            polygons_color = 'black'
        else:
            raise NotImplementedError
        
        ax.set_facecolor(background_color)
        for polygon in polygons_to_plot:
            v = polygon.vertices().transpose()
            hull = ConvexHull(v)
            plt.fill(*(v[hull.vertices].transpose()), 
                     facecolor=polygons_color,
                     edgecolor='blue')
        plt.xlim(self.bounds[0])
        plt.ylim(self.bounds[1])
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    
    def plot_convex_regions(self) -> None:
        fig, ax = plt.subplots()

        ax.set_facecolor('white')
        # plot obstacles
        for polygon in self.obstacles_vpolytopes:
            v = polygon.vertices().transpose()
            hull = ConvexHull(v)
            plt.fill(*(v[hull.vertices].transpose()), 
                     facecolor='black')
        # plot convex regions
        for polygon in self.regions_vpolytopes:
            v = polygon.vertices().transpose()
            hull = ConvexHull(v)
            plt.fill(*(v[hull.vertices].transpose()), 
                     facecolor='white',
                     edgecolor='blue')
        plt.xlim(self.bounds[0])
        plt.ylim(self.bounds[1])
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    def plot_trajectory(self, start: np.ndarray, end: np.ndarray,
                        waypoints: np.ndarray, mode: str) -> None:
        fig, ax = plt.subplots()
        if mode == 'regions':
            polygons_to_plot = self.regions_vpolytopes
            background_color = 'black'
            polygons_color = 'white'
        elif mode == 'obstacles':
            polygons_to_plot = self.obstacles_vpolytopes
            background_color = 'white'
            polygons_color = 'black'
        else:
            raise NotImplementedError
        
        # plot environment
        ax.set_facecolor(background_color)
        for polygon in polygons_to_plot:
            v = polygon.vertices().transpose()
            hull = ConvexHull(v)
            plt.fill(*(v[hull.vertices].transpose()),
                     facecolor=polygons_color)
        plt.xlim(self.bounds[0])
        plt.ylim(self.bounds[1])
        ax.set_aspect('equal', adjustable='box')

        # plot trajectory
        plt.plot(*start, 'go', mfc='none')
        plt.plot(*end, 'gx')
        plt.plot(*(waypoints.transpose()), 'b')

        plt.show()

    def to_img(self, 
               position: np.ndarray=None,
               start: np.ndarray=None,
               end: np.ndarray=None,
               shape: np.ndarray=np.array([64, 64])) -> np.ndarray:
        # plot environment
        fig, ax = plt.subplots()
        ax.set_facecolor('white')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        for polygon in self.obstacles_vpolytopes:   
            v = polygon.vertices().transpose()
            hull = ConvexHull(v)
            plt.fill(*(v[hull.vertices].transpose()),
                     facecolor='black')
        
        # plot the position if it is not None
        if position is not None:
            plt.plot(*position, 'ro', markersize=4)
        if start is not None:
            assert end is not None
            plt.plot(*start, 'go', mfc='none')
            plt.plot(*end, 'gx')
        
        plt.xlim(self.bounds[0])
        plt.ylim(self.bounds[1])
        ax.set_aspect('equal', adjustable='box')

        # Save the figure as a RGB numpy array without axis and with no padding
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Crop and resize the image
        # Note: This is a hacky solution! But I haven't found a better
        # way to remove the border. Other methods require saving to png
        # first, which would result in unnecessary I/O operations!

        # Find crop borders
        if self.left_crop is None:
            RGB_WHITE = np.array([255, 255, 255])
            vertical_midpoint = int(data.shape[0]/2)
            horizontal_midpoint = int(data.shape[1]/2)
            # find left crop
            for i in range(data.shape[1]):
                if not np.array_equal(data[vertical_midpoint, i, :], 
                                      RGB_WHITE):
                    self.left_crop = i
                    break
            # find right crop
            for i in range(data.shape[1]-1, -1, -1):
                if not np.array_equal(data[vertical_midpoint, i, :], 
                                      RGB_WHITE):
                    self.right_crop = i+1
                    break
            # find top crop
            for i in range(data.shape[0]):
                if not np.array_equal(data[i, horizontal_midpoint, :],
                                      RGB_WHITE):
                    self.top_crop = i
                    break
            # find bot crop
            for i in range(data.shape[0]-1, -1, -1):
                if not np.array_equal(data[i, horizontal_midpoint, :],
                                      RGB_WHITE):
                    self.bot_crop = i+1
                    break

        data = data[self.top_crop:self.bot_crop, 
                    self.left_crop:self.right_crop, :]
        data = cv2.resize(data, shape)
            
        plt.close()
        return data
    
    def animate_trajectory(self, 
                           waypoints: np.ndarray, 
                           start: np.ndarray=None, 
                           end: np.ndarray=None,
                           image_size: np.ndarray=np.array([64, 64]),
                           dt: float=0.1,
                           repeat: bool=True) -> None:
        """
        Animates a trajectory on the maze.

        waypoints: the waypoints of the trajectory
        start: the start point of the trajectory
        end: the end point of the trajectory
        dt: the time between frames in seconds
        """
        fig, ax = plt.subplots()
        num_frames = len(waypoints)
        def animate(i):
            plt.imshow(self.to_img(waypoints[i], start, end, shape=image_size))
        animation = FuncAnimation(fig, animate, frames=num_frames, 
                                interval=dt*1000, repeat=repeat)
        plt.show()



    """ Helper functions """
    def is_trajectory_success(self, start: np.ndarray, end: np.ndarray,
                              waypoints: np.ndarray, eps: float=0.1) -> bool:
        # check for collisions
        for waypoint in waypoints:
            if self.in_collision(waypoint, mode='obstacles'):
                return False
        # check starting condition
        if np.linalg.norm(waypoints[0]-start) > eps:
            return False
        # check ending condition
        if np.linalg.norm(waypoints[-1]-end) > eps:
            return False
        # passed all checks
        return True

    def in_collision(self, x: np.ndarray, mode: str='regions') -> bool:
        if mode == 'regions':
            for region in self.regions:
                if np.all(region.A() @ x <= region.b()):
                    return False
            return True
        elif mode == 'obstacles':
            for obstacle in self.obstacles:
                if np.all(obstacle.A() @ x <= obstacle.b()):
                    return True
            return False
        elif mode == 'padded_obstacles':
            for obstacle in self.padded_obstacles:
                if np.all(obstacle.A() @ x <= obstacle.b()):
                    return True
            return False
        else:
            raise NotImplementedError
        
    def get_planes(self, v_obstacle: VPolytope) -> Dict[str, float]:
        v = v_obstacle.vertices()
        return {
            'left': np.round(np.min(v[0]), 1),
            'right': np.round(np.max(v[0]), 1),
            'bot': np.round(np.min(v[1]), 1),
            'top': np.round(np.max(v[1]), 1)
        }

    def planes_to_hpolyhedron(self, planes: Dict[str, float]) -> HPolyhedron:
        for key, value in planes.items():
            planes[key] = np.round(value, 1)
        vertices = np.array([[planes['left'], planes['left'], planes['right'], planes['right']],
                             [planes['bot'], planes['top'], planes['bot'], planes['top']]])
        return HPolyhedron(VPolytope(vertices))

    def convert_to_vpolytope(self, polyhedrons: Polyhedrons) -> List[VPolytope]:
        """Converts list of HPolyhedrons to a list of VPolytopes
        
        Note: rounds vertices to nearest 10th (as per assumption)
        - This is done to avoid numerical issues when constructing the regions
        """
        v_polytopes = []
        for polyhedron in polyhedrons:
            # round vertices to powers of 10
            v = VPolytope(polyhedron).vertices()
            v_polytopes.append(VPolytope(np.round(v, 1)))
        return v_polytopes

    def get_padded_obstacles(self, obstacles_vpolytopes, padding, bounds) -> Polyhedrons:
        if padding <= 0.0:
            return self.obstacles
        
        padded_obstacles = []
        for v_obstacle in obstacles_vpolytopes:
            planes = self.get_planes(v_obstacle)
            planes['left'] = max(planes['left']-padding, bounds[0][0])
            planes['right'] = min(planes['right']+padding, bounds[0][1])
            planes['bot'] = max(planes['bot']-padding, bounds[1][0])
            planes['top'] = min(planes['top']+padding, bounds[1][1])
            padded_obstacles.append(self.planes_to_hpolyhedron(planes))
        return padded_obstacles

def print_binary_img(binary_img):
    for row in binary_img:
        for item in row:
            c = ' '
            if item == 1:
                c = '#'
            if item == 2:
                c = 'R'
            print(c, end=' ')
        print()


if __name__ == '__main__':
    """Tests for MazeEnvironment"""
    import data_generation.maze.gcs_utils as gcs_utils
    obstacles = gcs_utils.create_test_box_env()
    bounds = np.array([[0, 5], [0, 5]])

    maze_env = MazeEnvironment(bounds, obstacles=obstacles, 
                               obstacle_padding=0.1)
    binary_img = maze_env.get_binary_maze_representation(np.array([2.5, 2.5]))
    
    print()
    print_binary_img(binary_img)
    print()

    maze_env.plot_convex_regions()

    np_img = maze_env.to_img(
        position=np.array([2.5, 2.5])
    )
    
    newfig, ax = plt.subplots()
    ax.imshow(np_img)
    plt.show()

    traj = None
    while traj is None:
        start = maze_env.sample_start_point()
        end = maze_env.sample_end_point()
        traj = gcs_utils.run_gcs(maze_env.regions, start, end)
    
    waypoints = gcs_utils.composite_trajectory_to_array(traj).transpose()
    maze_env.plot_trajectory(start, end, waypoints, mode='obstacles')
    print(maze_env.is_trajectory_success(start, end, waypoints))
    maze_env.animate_trajectory(waypoints, dt=0.1, repeat=True)