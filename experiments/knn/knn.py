import torch
import numpy as np
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

@dataclass
class Neighbor:
    index: int
    distance: float
    dataset_index: int

def compute_knn(
    data: torch.Tensor, 
    query: torch.Tensor, 
    k: int,
    sqrt_Q: torch.Tensor,
) -> tuple:
    """
    Finds the k nearest neighbors for a query vector in the data.

    Args:
        data (torch.Tensor): The data tensor of shape (num_data_points, shape).
        query (torch.Tensor): The query vector of shape (shape,).
        k (int): The number of nearest neighbors to find.
        sqrt_Q (torch.Tensor): The square root of the Q matrix (quadratic norm)

    Returns:
        indices (torch.Tensor): Indices of the k nearest neighbors.
        distances (torch.Tensor): Distances to the k nearest neighbors.
    """
    # Ensure the query is a 2D tensor
    query = query.unsqueeze(0)  # Shape (1, shape)
    
    # Calculate the |.|_Q distances (|x^T*Q*x|_2)
    diff = data - query
    if torch.allclose(sqrt_Q, torch.eye(sqrt_Q.shape[0])):
        distances = torch.norm(diff, dim=1)
    else:
        distances = torch.norm(diff @ sqrt_Q, dim=1)
    
    # Get the indices of the top k smallest distances
    distances, indices = torch.topk(distances, k, largest=False)
    return indices, distances

def trajectory_knn(
    datasets: List[torch.Tensor],
    trajectory: torch.Tensor,
    k: int,
    sqrt_Q: torch.Tensor,
):
    num_datasets = len(datasets)
    # indexing order: [dataset_index][query_index][neighbor_index]
    nearest_neighbors_per_dataset = []

    for dataset_idx, dataset in enumerate(datasets):
        dataset_nearest_neighbors = []
        for query in tqdm(trajectory, desc=f"kNN w.r.t dataset {dataset_idx}"):
            query = query.flatten()
            indices, distances = compute_knn(dataset, query, k, sqrt_Q)
            neighbors = [Neighbor(index=indices[j].item(), distance=distances[j].item(), dataset_index=dataset_idx) for j in range(k)]
            dataset_nearest_neighbors.append(neighbors)
        nearest_neighbors_per_dataset.append(dataset_nearest_neighbors)
    
    # indexing order: [query_index][neighbor_index]
    nearest_neighbors = []
    for query_idx in range(len(trajectory)):
        candidates = []
        for dataset_idx in range(num_datasets):
            candidates.extend(nearest_neighbors_per_dataset[dataset_idx][query_idx])
        candidates.sort(key=lambda x: x.distance)
        nearest_neighbors.append(candidates[:k])
    
    return nearest_neighbors

def neighbors_to_dataset_indices(nearest_neighbors: List[List[Neighbor]]):
    dataset_indices = []
    n = len(nearest_neighbors)
    k = len(nearest_neighbors[0])

    for i in range(k):
        dataset_indices.append([nearest_neighbors[j][i].dataset_index for j in range(n)])

    # indexing order: [neighbor_index][query_index]
    return np.array(dataset_indices).astype(np.int32)

def neighbors_to_dataset_intensity(nearest_neighbors: List[List[Neighbor]], target_index: int=1):
    intensity = []
    k = 1.0*len(nearest_neighbors[0])
    for neighbors in nearest_neighbors:
        intensity.append(sum(neighbor.dataset_index == target_index for neighbor in neighbors) / k)

    # indexing order: [query_index]
    return np.array(intensity)