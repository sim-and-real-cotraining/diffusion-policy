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

def create_test_box_env():
    vertices_1 = np.array([[1.0, 1.0, 2.0, 2.0],
                           [3.0, 4.5, 3.0, 4.5]])
    region_1 = HPolyhedron(VPolytope(vertices_1))

    vertices_2 = np.array([[2.5, 2.5, 3.0, 3.0],
                           [1.0, 1.5, 1.0, 1.5]])
    region_2 = HPolyhedron(VPolytope(vertices_2))

    vertices_3 = np.array([[3.0, 3.0, 4.5, 4.5],
                           [2.5, 3.5, 2.5, 3.5]])
    region_3 = HPolyhedron(VPolytope(vertices_3))

    vertices_4 = np.array([[0.5, 0.5, 1.5, 1.5],
                           [1.0, 1.5, 1.0, 1.5]])
    region_4 = HPolyhedron(VPolytope(vertices_4))

    vertices_5 = np.array([[1.2, 1.2, 2.7, 2.7],
                           [0.5, 1.0, 0.5, 1.0]])
    region_5 = HPolyhedron(VPolytope(vertices_5))

    vertices_6 = np.array([[3.0, 3.0, 4.0, 4.0],
                           [4.0, 4.5, 4.0, 4.5]])
    region_6 = HPolyhedron(VPolytope(vertices_6))

    vertices_7 = np.array([[3.7, 3.7, 4.2, 4.2],
                           [0.5, 4.5, 0.5, 4.5]])
    region_7 = HPolyhedron(VPolytope(vertices_7))

    vertices_8 = np.array([[4.7, 4.7, 5.0, 5.0],
                            [1.0, 1.5, 1.0, 1.5]])
    region_8 = HPolyhedron(VPolytope(vertices_8))

    vertices_9 = np.array([[0.7, 1.5, 0.7, 1.5],
                           [2.5, 2.5, 3.5, 3.5]])
    region_9 = HPolyhedron(VPolytope(vertices_9))

    vertices_10 = np.array([[2.0, 2.3, 2.0, 2.3],
                           [2.7, 2.7, 3.5, 3.5]])
    region_10 = HPolyhedron(VPolytope(vertices_10))

    regions_ = [region_1, region_2, region_3, region_4, 
                region_5, region_6, region_7, region_8,
                region_9, region_10]
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

    # start_time = time.time()
    [traj, result] = gcs.SolvePath(source, target, options)
    # if result.is_success():
    #     print(f"Solved in {time.time()-start_time}s")
    # else:
    #     print("Failure.")
    return traj if result.is_success() else None


def plot_environment(regions, start, goal, waypoints, num_points=100):
    # Given a list of HPolyhedrons in regions, plot the environment using matplotlib
    fig, ax = plt.subplots()
    ax.set_facecolor("black")
    for region in regions:
        v = VPolytope(region).vertices().transpose()
        hull = ConvexHull(v)
        plt.fill(*(v[hull.vertices].transpose()), color='white')

    plt.plot(*start, 'gx', )
    plt.plot(*goal, 'gx')
    plt.plot(*waypoints, 'b')

    plt.xlim([0, 5])
    plt.ylim([0, 5])
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def save_trajectory_plot(filename,
                         regions, bounds, start, goal, waypoints, 
                         velocity_bounds=np.array([[-1,1],[-1,1]]),
                         collision_indices=[1, 4],
                         dt=0.1,
                         gcs_regions=None):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # plot obstacles
    ax1.set_facecolor("black")
    for region in regions:
        v = VPolytope(region).vertices().transpose()
        hull = ConvexHull(v)
        ax1.fill(*(v[hull.vertices].transpose()), color='white')
    if gcs_regions is not None:
        for region in gcs_regions:
            v = VPolytope(region).vertices().transpose()
            hull = ConvexHull(v)
            ax1.fill(*(v[hull.vertices].transpose()), 
                     color='white',
                     edgecolor='green',)

    # plot trajectory
    ax1.plot(*start, 'go', mfc='none')
    ax1.plot(*goal, 'gx')
    ax1.plot(*waypoints, 'b')

    # plot collisions
    colliding_points = waypoints[:,collision_indices]
    ax1.scatter(*colliding_points, color='r', s=4, zorder=2)

    ax1.set_xlim(bounds[0])
    ax1.set_ylim(bounds[1])
    ax1.set_aspect('equal', adjustable='box')

    # compute velocities
    velocities_x = [0]
    velocities_y = [0]
    for i in range(1, waypoints.shape[1]):
        velocities_x.append((waypoints[0,i] - waypoints[0,i-1])/dt)
        velocities_y.append((waypoints[1,i] - waypoints[1,i-1])/dt)
    
    T = waypoints.shape[1]*dt
    t = np.linspace(0, T-dt, waypoints.shape[1])

    # plot velocities
    ax2.axhline(velocity_bounds[0,0], color='orange', linestyle='--')
    ax2.axhline(velocity_bounds[0,1], color='orange', linestyle='--')
    ax2.axhline(velocity_bounds[1,0], color='blue', linestyle='--')
    ax2.axhline(velocity_bounds[1,1], color='blue', linestyle='--')
    ax2.plot(t, velocities_x, 'orange')
    ax2.plot(t, velocities_y, 'blue')
    fig.set_figwidth(2.5)
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

def composite_trajectory_to_array(traj, dt=0.1):
    num_waypoints = int(traj.end_time() / dt) + 1       # +1 to make sure we get the last point
    traj_points = np.zeros((2, num_waypoints))
    for i in range(num_waypoints):
        traj_points[:,i] = traj.value(i*dt).reshape(2,)
    return traj_points

def check_velocity_bounds(waypoints, velocity_bounds, dt):
    # compute velocities
    velocities_x = []
    velocities_y = []
    for i in range(1, waypoints.shape[1]):
        velocities_x.append((waypoints[0,i] - waypoints[0,i-1])/dt)
        velocities_y.append((waypoints[1,i] - waypoints[1,i-1])/dt)
    velocities_x = np.array(velocities_x)
    velocities_y = np.array(velocities_y)

    if np.any(velocities_x < velocity_bounds[0,0]) or np.any(velocities_x > velocity_bounds[0,1]):
        return False
    if np.any(velocities_y < velocity_bounds[1,0]) or np.any(velocities_y > velocity_bounds[1,1]):
        return False
    return True

def get_colliding_indices(regions, waypoints):
    colliding_indices = []
    for i, point in enumerate(waypoints.transpose()):
        # Check for high acceleration
        # if 0 < i < len(waypoints.transpose())-1:
        #     v = (waypoints[:,i] - waypoints[:,i-1]) / 0.1
        #     v_next = (waypoints[:,i+1] - waypoints[:,i]) / 0.1
        #     a = abs((v_next - v) / 0.1)
        #     if a[0] > 10 or a[1] > 10:
        #         colliding_indices.append(i)
        #         continue
        if in_collision(regions, point):
            colliding_indices.append(i)
    return colliding_indices

def in_collision(regions, point):
    """
    LEGACY: use MazeEnvironment.in_collision instead
    """
    for region in regions:
        if np.all(region.A() @ point <= region.b()):
            return False
    return True

def sample_collision_free_point(regions, bounds):
    """
    LEGACY: use MazeEnvironment.sample_collision_free_point instead
    """
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

def log_eval_results(filename,
                     passed_all_tests,
                     passed_collision_tests,
                     passed_velocity_tests,
                     failed_to_converge,
                     num_traj):
    pass_ratio = len(passed_all_tests)/num_traj
    collision_free_ratio = len(passed_collision_tests)/num_traj
    satisfies_velocity_bounds_ratio = len(passed_velocity_tests)/num_traj
    converged_ratio = (num_traj-len(failed_to_converge))/num_traj

    f = open(filename, "w")
    f.write("Evaluation Results\n-----------------\n")
    f.write(f"Passed All Tests: {100.0*pass_ratio:.2f}%\n")
    f.write(f"Collision Free: {100.0*collision_free_ratio:.2f}%\n")
    f.write(f"Satisfies Velocity Constraints: {100.0*satisfies_velocity_bounds_ratio:.2f}%\n")
    f.write(f"Converged: {100.0*converged_ratio:.2f}%\n")
    f.write(f"\n")

    f.write(f"Passed All Tests: {passed_all_tests}\n")
    f.write(f"Collision Free: {passed_collision_tests}\n")
    f.write(f"Satisfies Velocity Constraints: {passed_velocity_tests}\n")
    f.write(f"Failed To Converge: {failed_to_converge}\n")
    f.write(f"\n")

    f.write("See plots directory for visualizations.")
    f.close()


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
            pbar.update(1)

        # rebuild GCS object if solve times become too slow
        # (i.e. graph got too large)
        if time.time() - start_time > 0.75:
            gcs = GcsTrajectoryOptimization(n)
            free_space = gcs.AddRegions(regions, 3)
            gcs.AddVelocityBounds(np.array([-max_speed, -max_speed]), np.array([max_speed, max_speed]))
            gcs.AddPathContinuityConstraints(continuity_order)
            gcs.AddTimeCost()
            gcs.AddPathLengthCost()


    pbar.close()
    return data