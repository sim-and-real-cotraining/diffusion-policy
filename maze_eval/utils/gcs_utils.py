import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import logging
import tqdm
from multiprocessing import Pool
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from datetime import datetime

from pydrake.geometry.optimization import (HPolyhedron,
                                           VPolytope,
                                           Point,
                                           GraphOfConvexSetsOptions)
from pydrake.solvers import MosekSolver
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.common import configure_logging

# Initial test script for GCS
def create_env():
    # create environment
    vertices_1 = np.array([[0.4, 0.4, 0.0, 0.0],
                            [0.0, 5.0, 5.0, 0.0]])
    region_1 = HPolyhedron(VPolytope(vertices_1))

    vertices_2 = np.array([[0.4, 1.0, 1.0, 0.4],
                            [2.4, 2.4, 2.6, 2.6]])
    region_2 = HPolyhedron(VPolytope(vertices_2))

    vertices_3 = np.array([[1.4, 1.4, 1.0, 1.0],
                            [2.2, 4.6, 4.6, 2.2]])
    region_3 = HPolyhedron(VPolytope(vertices_3))

    vertices_4 = np.array([[1.4, 2.4, 2.4, 1.4],
                            [2.2, 2.6, 2.8, 2.8]])
    region_4 = HPolyhedron(VPolytope(vertices_4))

    vertices_5 = np.array([[2.2, 2.4, 2.4, 2.2],
                            [2.8, 2.8, 4.6, 4.6]])
    region_5 = HPolyhedron(VPolytope(vertices_5))

    vertices_6 = np.array([[1.4, 1., 1., 3.8, 3.8],
                            [2.2, 2.2, 0.0, 0.0, 0.2]])
    region_6 = HPolyhedron(VPolytope(vertices_6))

    vertices_7 = np.array([[3.8, 3.8, 1.0, 1.0],
                            [4.6, 5.0, 5.0, 4.6]])
    region_7 = HPolyhedron(VPolytope(vertices_7))

    vertices_8 = np.array([[5.0, 5.0, 4.8, 3.8, 3.8],
                            [0.0, 1.2, 1.2, 0.2, 0.0]])
    region_8 = HPolyhedron(VPolytope(vertices_8))

    vertices_9 = np.array([[3.4, 4.8, 5.0, 5.0],
                            [2.6, 1.2, 1.2, 2.6]])
    region_9 = HPolyhedron(VPolytope(vertices_9))

    vertices_10 = np.array([[3.4, 3.8, 3.8, 3.4],
                            [2.6, 2.6, 4.6, 4.6]])
    region_10 = HPolyhedron(VPolytope(vertices_10))

    vertices_11 = np.array([[3.8, 4.4, 4.4, 3.8],
                            [2.8, 2.8, 3.0, 3.0]])
    region_11 = HPolyhedron(VPolytope(vertices_11))

    vertices_12 = np.array([[5.0, 5.0, 4.4, 4.4],
                            [2.8, 5.0, 5.0, 2.8]])
    region_12 = HPolyhedron(VPolytope(vertices_12))

    regions_ = [region_1, region_2, region_3, region_4, region_5, region_6,
                region_7, region_8, region_9, region_10, region_11, region_12]
    return regions_

def run_gcs(regions, start, goal):
    n = 2
    max_speed = 1.0
    continuity_order = 1
    gcs = GcsTrajectoryOptimization(n)

    # Build graph
    free_space = gcs.AddRegions(regions, 3)
    source = gcs.AddRegions([Point(start)], 0)
    target = gcs.AddRegions([Point(goal)], 0)

    gcs.AddEdges(source, free_space)
    gcs.AddEdges(free_space, target)

    # Cost & Constraints
    gcs.AddVelocityBounds(np.array([-max_speed, -max_speed]), np.array([max_speed, max_speed]))
    gcs.AddPathContinuityConstraints(continuity_order)
    gcs.AddTimeCost()
    gcs.AddPathLengthCost()

    options = GraphOfConvexSetsOptions()
    options.max_rounded_paths = 3

    # start = time.time()
    [traj, result] = gcs.SolvePath(source, target, options)
    # if result.is_success():
    #     print(f"Solved in {time.time()-start}s")
    # else:
    #     print("Failure.")
    return traj if result else None


def plot_environment(regions, start, goal, waypoints, num_points=100):
    # Given a list of HPolyhedrons in regions, plot the environment using matplotlib
    fig, ax = plt.subplots()
    ax.set_facecolor("black")
    for region in regions:
        v = VPolytope(region).vertices().transpose()
        hull = ConvexHull(v)
        plt.fill(*(v[hull.vertices].transpose()), color='white')

    plt.plot(*start, 'rx', )
    plt.plot(*goal, 'rx')
    plt.plot(*waypoints, 'b')

    plt.xlim([0, 5])
    plt.ylim([0, 5])
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def composite_trajectory_to_array(traj, dt=0.1):
    num_waypoints = int(traj.end_time() / dt) + 1       # +1 to make sure we get the last point
    traj_points = np.zeros((2, num_waypoints))
    for i in range(num_waypoints):
        traj_points[:,i] = traj.value(i*dt).reshape(2,)
    return traj_points

def in_collision(regions, point):
    for region in regions:
        if np.all(region.A() @ point <= region.b()):
            return False
    return True

def sample_collision_free_point(regions, bounds):
    while True:
        sample = np.random.uniform(bounds[:,0], bounds[:,1])
        if not in_collision(regions, sample):
            return sample


def collision_free_grid(regions, bounds, points_per_axis):
    X = np.linspace(bounds[0,0], bounds[0,1], points_per_axis)
    Y = np.linspace(bounds[1,0], bounds[1,1], points_per_axis)

    collision_free_points = []
    for x in X:
        for y in Y:
            point = np.array([x, y])
            collision_free_points.append(point) if not in_collision(regions, point) else None

    return np.array(collision_free_points)

def generate_data(regions, bounds, N=10):
    # build GCS
    n = 2
    max_speed = 1.0
    continuity_order = 1
    gcs = GcsTrajectoryOptimization(n)

    # Build graph
    free_space = gcs.AddRegions(regions, 3)

    # Cost & Constraints
    gcs.AddVelocityBounds(np.array([-max_speed, -max_speed]), np.array([max_speed, max_speed]))
    gcs.AddPathContinuityConstraints(continuity_order)
    gcs.AddTimeCost()
    gcs.AddPathLengthCost()

    options = GraphOfConvexSetsOptions()
    options.max_rounded_paths = 3

    # generate seed for thread
    seed = int((datetime.now().timestamp() % 1) * 1e6)
    np.random.seed(seed)

    data = []
    pbar = tqdm.tqdm(total=N)
    while len(data) < N:
        start_time = time.time()
        start = sample_collision_free_point(regions, bounds)
        goal = sample_collision_free_point(regions, bounds)
        source = gcs.AddRegions([Point(start)], 0)
        target = gcs.AddRegions([Point(goal)], 0)
        gcs.AddEdges(source, free_space)
        gcs.AddEdges(free_space, target)

        [traj, result] = gcs.SolvePath(source, target, options)
        if result.is_success():
            waypoints = composite_trajectory_to_array(traj)
            data.append((start, goal, waypoints))

        # rebuild GCS object if solve times become too slow
        # (i.e. graph got too large)
        if time.time() - start_time > 0.75:
            gcs = GcsTrajectoryOptimization(n)
            free_space = gcs.AddRegions(regions, 3)
            gcs.AddVelocityBounds(np.array([-max_speed, -max_speed]), np.array([max_speed, max_speed]))
            gcs.AddPathContinuityConstraints(continuity_order)
            gcs.AddTimeCost()
            gcs.AddPathLengthCost()

        pbar.update(1)

    pbar.close()
    return data