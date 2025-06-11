import numpy as np
from tqdm import tqdm
from data_generation.motion_planners.common import (
    TreeNode, Tree, _euclidean_distance, KDTreePayload
)
import kdtree

class BaseRRT:
    def __init__(self, source, max_step_size=0.1):
        self.source = source
        self.RRT_tree = Tree(TreeNode(source))
        self.vertices = [self.RRT_tree.root] # easy indexing into all vertices
        self.max_step_size = max_step_size
        self.kdtree = kdtree.create([KDTreePayload(source, self.RRT_tree.root)])

    def sample_free(self):
        """
        Returns a random point in the free space.
        """
        raise NotImplementedError()
    
    def is_free(self, q):
        """
        Returns True if the configuration q is in the free space.
        """
        raise NotImplementedError()

    def find_nearest(self, q, 
                     distance_metric=_euclidean_distance) -> TreeNode:
        """
        Returns the nearest node to configuration q in the tree.

        Args:
            q: a configuration in the configuration space
            distance_metric: a function that takes in two 
            configurations and returns the distance between them

        If no custom distance function is provided, the default
        this function will use a kd-tree to find the nearest node.

        If a custom distance function is provided, brute force
        is used instead of a kd-tree which runs in O(V) time.
        Using this method, the RRT algorithm runs in O(V^2) time.
        To improve runtime, override this method with something
        like a kd-tree.
        """
        nearest_node = None
        nearest_distance = float('inf')
        # kd tree
        if distance_metric is _euclidean_distance:
            kd_nearest, _ = self.kdtree.search_nn(q)
            nearest_node = kd_nearest.data.data
            nearest_distance = _euclidean_distance(nearest_node.value, q)
        # brute force
        else:
            for vertex in self.vertices:
                distance = distance_metric(q, vertex.value)
                if distance < nearest_distance:
                    nearest_node = vertex
                    nearest_distance = distance
        return nearest_node, nearest_distance
    
    def steer(self, q, nearest_node, distance):
        """
        Returns a new configuration by moving from nearest_node to q
        in the direction of q, with a maximum distance of step_size.
        """
        step_size = min(self.max_step_size, distance)
        return nearest_node.value + \
            (q - nearest_node.value) * step_size / distance
    
    def add_vertex(self, q):
        """
        Adds a new vertex to the tree with value q.
        Returns new_node if success, None otherwise.
        """
        nearest_node, distance = self.find_nearest(q)
        q_new = self.steer(q, nearest_node, distance)

        # collision check
        if not self.is_free(q_new):
            return None
        if not self._obstacle_free(nearest_node.value, q_new):
            return None
        
        # collision-free, add new node to the tree
        new_node = TreeNode(q_new, nearest_node)
        self.RRT_tree.add_node(new_node, nearest_node)
        self.vertices.append(new_node)
        self.kdtree.add(KDTreePayload(q_new, new_node))
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
    
    def grow_to_goal(self, q_goal, 
                     max_samples=-1,
                     distance_metric = _euclidean_distance,
                     num_shortcut_attempts: int=0):
        """
        Grows the RRT tree to q_goal. Returns path if successful, None otherwise.
        """
        # check if a node already exists
        assert self.is_free(q_goal)
        nearest_node, distance = self.find_nearest(q_goal, distance_metric)
        if distance < self.max_step_size and self._obstacle_free(nearest_node.value, q_goal):
            return self.find_path(q_goal)
        
        # no node close to goal. continue growing
        found_path = False
        i = 0
        while max_samples < 0 or i < max_samples:
            new_node = self.sample_and_add_vertex()
            if _euclidean_distance(new_node.value, q_goal) < self.max_step_size:
                found_path = True
                break
            i += 1
        
        if not found_path:
            return None
        else:
            return self.find_path(q_goal, num_shortcut_attempts)
        
    def find_path(self, q_goal, num_shortcut_attempts: int=0):
        """
        Returns a path from the root to q_goal.
        If no path is found, return None
        """
        # check for collisions
        if not self.is_free(q_goal):
            return None
        nearest_node, distance = self.find_nearest(q_goal)
        if not self._obstacle_free(nearest_node.value, q_goal, 
                               n=int(distance // 0.1)):
            return None
        
        # passed collision checks: find path
        path = self.RRT_tree.find_path_to_root(nearest_node)
        path.append(q_goal)
        # shortcut path
        if num_shortcut_attempts > 0:
            path = self.shortcut_path(path, num_shortcut_attempts)
        return path
    
    def shortcut_path(self, path, num_attempts=100):
        """
        Shortcuts the path using the straight line path between
        two configurations.
        """
        # try straight line path
        src, dst = path[0], path[-1]
        straight_line_dist = _euclidean_distance(src, dst)
        if self._obstacle_free(src, dst, int(straight_line_dist // 0.05)):
            return [src, dst]

        # try random shortcuts
        for _ in range(int(num_attempts)):
            if len(path) < 3:
                return path
            # sample indices i and j st.
            # 0 <= i < j < len(path) and j - i > 1
            i = np.random.randint(0, len(path)-2)
            j = np.random.randint(i+2, len(path))
            num_points = int(_euclidean_distance(path[i], path[j]) // 0.1)
            if self._obstacle_free(path[i], path[j], num_points):
                path = path[:i+1] + path[j:]
        return path
    
    def visualize(self, path=None):
        """
        Visualizes the tree and the path.
        """
        raise NotImplementedError
    
    def _obstacle_free(self, q1, q2, n=50):
        """
        Returns True if the straight line path between q1 and q2 is obstacle-free.
        """
        for i in range(int(n)):
            q_intermediate = q1 + (q2 - q1) * (i + 1) / (n+2)
            if not self.is_free(q_intermediate):
                return False
        return True
    
    def reset(self):
        # garbage collection should happen naturally
        self.RRT_tree = Tree(TreeNode(self.source))
        self.vertices = [self.RRT_tree.root] # easy indexing into all vertices
        self.kdtree = kdtree.create([KDTreePayload(self.source, self.RRT_tree.root)])