import numpy as np

class TreeNode:
    def __init__(self, value, parent = None):
        assert isinstance(parent, TreeNode) or parent is None
        self.value = value
        self.parent = parent
        self.cost = 0 # used in RRT*
        self.children = []

class Tree:
    def __init__(self, root: TreeNode):
        self.root = root

    def add_node(self, node: TreeNode, parent: TreeNode):
        parent.children.append(node)
        node.parent = parent

    def find_path_to_root(self, node: TreeNode):
        path = []
        dummy_node = TreeNode(node.value, node.parent) # dummy node to avoid modifying the tree
        while dummy_node is not None:
            path.append(dummy_node.value)
            dummy_node = dummy_node.parent
        path.reverse()
        return path
    
# payload class for kdtree. emulates a tuple
class KDTreePayload(object):
    def __init__(self, q, data):
        self.q = q
        self.data = data
    
    def __len__(self):
        return len(self.q)
    
    def __getitem__(self, i):
        return self.q[i]
    
    def __repr__(self):
        return f'({self.q}, {self.data})'
    
def _euclidean_distance(a, b):
    """
    Returns the Euclidean distance between a and b.
    """
    return np.linalg.norm(a - b)
