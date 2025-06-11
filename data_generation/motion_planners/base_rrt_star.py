import numpy as np
import math
from tqdm import tqdm
from data_generation.motion_planners.common import (
    TreeNode, Tree, _euclidean_distance, KDTreePayload
)
from data_generation.motion_planners.base_rrt import BaseRRT
import kdtree

class BaseRRTStar(BaseRRT):
    def __init__(self, source, max_step_size=0.1):
        super().__init__(source, max_step_size)
        self.k_RRT = 6 # any number greater than 2e will work

    def k_nearest_neighbors(self, q, k, 
                            distance_metric=_euclidean_distance):
        if distance_metric is _euclidean_distance:
            knn = self.kdtree.search_knn(q, k)
            distances = [distance_metric(kd_node.data.q, q) for kd_node, dist in knn]
            return [(knn[i][0].data.data, distances[i]) for i in range(len(knn))]
        else:
            # Brute force solution
            raise NotImplementedError()

    def add_vertex(self, q):
        """
        Adds a new vertex to the tree with value q
        and performs rewiring.
        Returns new_node if success, None otherwise.
        """
        nearest_node, distance = self.find_nearest(q)
        q_new = self.steer(q, nearest_node, distance)

        # Collision check
        if not self.is_free(q_new):
            return None
        if not self._obstacle_free(nearest_node.value, q_new, 50):
            return None
        
        # Collision-free, add new node to the tree
        k = self._get_k()
        nearest_neighbors = self.k_nearest_neighbors(q_new, k)
        # Connect along a minimum-cost path
        x_new_parent = nearest_node
        min_cost = nearest_node.cost + _euclidean_distance(nearest_node.value, q_new)
        for node, distance in nearest_neighbors:
            if not self._obstacle_free(node.value, q_new, distance // 0.05):
                continue
            new_cost = node.cost + distance
            if new_cost < min_cost:
                x_new_parent = node
                min_cost = new_cost

        # add new node to RRT
        new_node = TreeNode(q_new, x_new_parent)
        new_node.cost = min_cost
        self.RRT_tree.add_node(new_node, x_new_parent)
        self.vertices.append(new_node)
        self.kdtree.add(KDTreePayload(q_new, new_node))

        # rewire the tree
        for node, distance in nearest_neighbors:
            if not self._obstacle_free(q_new, node.value, distance // 0.05):
                continue
            new_cost = new_node.cost + distance
            if new_cost < node.cost:
                old_parent = node.parent
                # old parent is None iff node is root
                # the lowest cost path to the root is [root]
                # which means no rewiring is necessary
                if old_parent is not None:
                    old_parent.children.remove(node)
                    node.parent = new_node
                    new_node.children.append(node)
                    node.cost = new_cost
        
        return new_node
        
    def sample_and_add_vertex(self):
        """
        Samples a new configuration and adds it to the tree.
        Returns True if success, False otherwise.
        """
        new_node = None
        while new_node is None:
            q = self.sample_free()
            new_node = self.add_vertex(q)
        return new_node

    def grow(self, N: int):
        """
        Grows the RRT tree.
        """
        for _ in tqdm(range(N), desc='Growing RRT'):
            self.sample_and_add_vertex()

    def find_path(self, q_goal, num_shortcut_attempts: int=0):
        # check for collisions
        if not self.is_free(q_goal):
            return None
        k = self._get_k()
        nearest_nodes = self.k_nearest_neighbors(q_goal, k)

        parent_node = None
        parent_cost = float('inf')
        for node, distance in nearest_nodes:
            if not self._obstacle_free(node.value, q_goal, int(distance // 0.05)):
                continue
            new_cost = node.cost + distance
            if new_cost < parent_cost:
                parent_node = node
                parent_cost = new_cost
        
        if parent_node == None:
            return None
        
        path = self.RRT_tree.find_path_to_root(parent_node)
        path.append(q_goal)
        if num_shortcut_attempts > 0:
            path = self.shortcut_path(path, num_shortcut_attempts)
        return path
    
    def _get_k(self):
        k = self.k_RRT * math.log(len(self.vertices))
        return int(math.floor(k+1))