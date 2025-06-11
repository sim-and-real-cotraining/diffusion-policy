import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import wasserstein_distance_nd
import time
from scipy.sparse import csc_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gaussian kernel for MMD computation
def gaussian_kernel_torch(x, y, sigma=1.0):
    dist = torch.cdist(x, y) ** 2
    return torch.exp(-dist / (2 * sigma ** 2))

# Torch-based MMD computation with batching
def compute_mmd(file1, file2, sigma=1.0, batch_size=1024):
    # Load numpy arrays and convert to torch tensors
    data1 = torch.tensor(np.load(file1), device=device, dtype=torch.float32)
    data2 = torch.tensor(np.load(file2), device=device, dtype=torch.float32)

    # Ensure dimensions match
    assert data1.shape[1] == data2.shape[1], "Datasets must have the same feature dimension."

    n1, n2 = data1.size(0), data2.size(0)

    # Compute k_xx in batches
    k_xx_sum = 0.0
    count_xx = 0
    for i in range(0, n1, batch_size):
        batch1 = data1[i:i+batch_size]
        for j in range(0, n1, batch_size):
            batch2 = data1[j:j+batch_size]
            k_xx = gaussian_kernel_torch(batch1, batch2, sigma)
            if i == j:
                k_xx.fill_diagonal_(0)
            k_xx_sum += k_xx.sum().item()
            count_xx += k_xx.numel() - (k_xx.size(0) if i == j else 0)
    k_xx_mean = k_xx_sum / count_xx

    # Compute k_yy in batches
    k_yy_sum = 0.0
    count_yy = 0
    for i in range(0, n2, batch_size):
        batch1 = data2[i:i+batch_size]
        for j in range(0, n2, batch_size):
            batch2 = data2[j:j+batch_size]
            k_yy = gaussian_kernel_torch(batch1, batch2, sigma)
            if i == j:
                k_yy.fill_diagonal_(0)
            k_yy_sum += k_yy.sum().item()
            count_yy += k_yy.numel() - (k_yy.size(0) if i == j else 0)
    k_yy_mean = k_yy_sum / count_yy

    # Compute k_xy in batches
    k_xy_sum = 0.0
    count_xy = 0
    for i in range(0, n1, batch_size):
        batch1 = data1[i:i+batch_size]
        for j in range(0, n2, batch_size):
            batch2 = data2[j:j+batch_size]
            k_xy = gaussian_kernel_torch(batch1, batch2, sigma)
            k_xy_sum += k_xy.sum().item()
            count_xy += k_xy.numel()
    k_xy_mean = k_xy_sum / count_xy

    # Compute MMD
    mmd_squared = k_xx_mean + k_yy_mean - 2 * k_xy_mean
    mmd = torch.sqrt(torch.clamp(torch.tensor(mmd_squared, device=device), min=0))

    return mmd.item()


def compute_wasserstein_distance(file1, file2, n):
    data1 = np.load(file1)
    data2 = np.load(file2)

    # TODO: cubic in n
    data1 = data1.reshape(data1.shape[0], -1)
    data2 = data2.reshape(data2.shape[0], -1)
    data1 = data1[:n]
    data2 = data2[:n]

    return wasserstein_distance_nd(data1, data2)

def compute_wasserstein_distance_drake(file1, file2, n, verbose=False):
    from pydrake.all import (
        MathematicalProgram,
        Solve,
    )
    data1 = np.load(file1)
    data2 = np.load(file2)
    
    # subsample to reduce k-means computation
    min_len = min(len(data1), len(data2))
    indices1 = np.random.choice(len(data1), size=min_len, replace=False)
    indices2 = np.random.choice(len(data2), size=min_len, replace=False)
    data1 = data1[indices1]
    data2 = data2[indices2]

    assert len(data1.shape) == 2 and len(data2.shape) == 2, "Datasets must be 2D."
    assert data1.shape[1] == data2.shape[1], "Datasets must have the same feature dimension."
    
    # kmeans clustering
    start = time.time()
    clusters_1, _ = kmeans(data1, n, device='cpu')
    clusters_2, _ = kmeans(data2, n, device='cpu')
    mu = compute_cluster_histograms(data1, clusters_1)
    nu = compute_cluster_histograms(data2, clusters_2)
    n, m = len(mu), len(nu)
    if verbose:
        print("Kmeans Time:", time.time() - start)

    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(n*m, "x")

    # Add cost
    start = time.time()
    c = torch.cdist(
        torch.tensor(clusters_1), 
        torch.tensor(clusters_2)
    ).numpy().flatten()
    cost = prog.AddLinearCost(c, x)
    if verbose:
        print("Time to add cost:", time.time() - start)

    # Add sum_j x_ij = mu_i constraints ()
    start = time.time()
    a = np.ones((1, m))
    for i in range(n):
        start_inner = time.time()
        prog.AddLinearEqualityConstraint(a, mu[i], x[i*m:(i+1)*m])
        if i == 0 and verbose:
            print("Time to add single mu constraint:", time.time() - start_inner)
    if verbose:
        print("Time to add mu constraints:", time.time() - start)
    
    start = time.time()
    a = np.ones((1, n))
    for j in range(m):
        start_inner = time.time()
        prog.AddLinearEqualityConstraint(a, nu[j], x[j::m])
        if j == 0 and verbose:
            print("Time to add single nu constraint:", time.time() - start_inner)
    if verbose:
        print("Time to add nu constraints:", time.time() - start)
    
    # x >= 0
    prog.AddBoundingBoxConstraint(0, float('inf'), x)
    print("Finished building program")
    
    start = time.time()
    result = Solve(prog)
    if verbose:
        print("Solve Time:", time.time() - start)
    return result.get_optimal_cost()

def compute_cluster_histograms(data, clusters):
    distances = np.linalg.norm(data[:, np.newaxis, :] - clusters[np.newaxis, :, :], axis=2)
    cluster_labels = np.argmin(distances, axis=1)
    histogram = np.zeros(clusters.shape[0])
    for label in cluster_labels:
        histogram[label] += 1
    return histogram / len(data)

def kmeans(data, k, max_iters=100, tol=1e-4, verbose=False, device='cuda'):
    """
    K-means algorithm using PyTorch with GPU acceleration.

    Args:
        data (torch.Tensor): Input data of shape (N, D), where N is the number of points and D is the dimensionality.
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Convergence tolerance based on centroid movement.
        verbose (bool): If True, print progress during iterations.
        device (str): Device to run the algorithm on ('cuda' for GPU, 'cpu' for CPU).

    Returns:
        centroids: Final cluster centroids of shape (k, D).
        labels: Cluster assignments of shape (N,).
    """
    # Move data to the specified device
    if type(data) is np.ndarray:
        data = torch.tensor(data, device=device)
    data = data.to(device)
    N, D = data.shape

    # Initialize centroids by randomly selecting k points from the data
    indices = torch.randperm(N, device=device)[:k]
    centroids = data[indices]

    for i in range(max_iters):
        # Compute pairwise distances (N, k)
        distances = torch.cdist(data, centroids, p=2)

        # Assign each data point to the nearest centroid (N,)
        labels = torch.argmin(distances, dim=1)

        # Compute new centroids (k, D)
        new_centroids = torch.stack([
            data[labels == cluster_idx].mean(dim=0)
            if torch.any(labels == cluster_idx) else centroids[cluster_idx]
            for cluster_idx in range(k)
        ])

        # Check for convergence (based on centroid movement)
        movement = torch.norm(new_centroids - centroids, dim=1).max().item()
        if verbose:
            print(f"Iteration {i+1}, Movement: {movement:.6f}")
        if movement < tol:
            break

        centroids = new_centroids

    # Move results back to CPU for further use
    return centroids.cpu().numpy(), labels.cpu().numpy()

# Main function
def main(pairs, sigma=1.0):
    results = []
    for file1, file2 in pairs:
        mmd = compute_mmd(file1, file2, sigma)
        results.append(mmd)

    # Print results in a readable format
    return np.array(results)

if __name__ == "__main__":
    no_gan_50_500_pairs = [
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_3_1_emb_10/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_3_1_emb_10/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_1_1_emb_10/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_1_1_emb_10/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_1_3_emb_10/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_1_3_emb_10/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_emb_10/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/50_500_emb_10/seed_0/sim_embeddings.npy'
        ),
    ]
    gan_50_500_pairs = [
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/50_500_3_1/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/50_500_3_1/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/50_500_1_1/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/50_500_1_1/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/50_500_1_3/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/50_500_1_3/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/50_500/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/50_500/seed_0/sim_embeddings.npy'
        ),
    ]
    no_gan_10_2000_pairs = [
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_3_1_emb_10/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_3_1_emb_10/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_1_1_emb_10/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_1_1_emb_10/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_1_3_emb_10/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_1_3_emb_10/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_emb_10/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/embedding_dim/10_2000_emb_10/seed_0/sim_embeddings.npy'
        ),
    ]
    gan_10_2000_pairs = [
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/10_2000_3_1/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/10_2000_3_1/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/10_2000_1_1/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/10_2000_1_1/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/10_2000_1_3/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/10_2000_1_3/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/10_2000/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/adversarial/gan_emb_dim/10_2000/seed_0/sim_embeddings.npy'
        ),
    ]
    no_gan_50_500_pairs_regular_emb = [
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/50_500_3_1/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/50_500_3_1/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/50_500_1_1/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/50_500_1_1/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/50_500_1_3/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/50_500_1_3/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/50_500/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/50_500/seed_0/sim_embeddings.npy'
        ),
    ]
    no_gan_10_2000_pairs_regular_emb = [
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/10_2000_3_1/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/10_2000_3_1/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/10_2000_1_1/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/10_2000_1_1/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/10_2000_1_3/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/10_2000_1_3/seed_0/sim_embeddings.npy'
        ),
        (
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/10_2000/seed_0/real_embeddings.npy',
            '/home/adam/workspace/gcs-diffusion/data/experiments/cotrain/scaled_mixing/10_2000/seed_0/sim_embeddings.npy'
        ),
    ]

    # print("Wasserstein Results:")
    # for i, pair in enumerate(no_gan_50_500_pairs):
    #     start = time.time()
    #     for n in [10, 100, 200, 400, 1000, 2000]:
    #         wasserstein_distance = compute_wasserstein_distance(pair[0], pair[1], n)
    #         print(f"Wasserstein Distance {i}: {wasserstein_distance:.4f}")
    #         print("Time:", time.time() - start)

    print("Wasserstein Results: no gan 50 500")
    for i, pair in enumerate(no_gan_50_500_pairs):
        num_trials = 1
        wds = np.zeros(num_trials)
        for trial in range(num_trials):
            wd = compute_wasserstein_distance_drake(
                pair[0], pair[1], 10000, verbose=True
            )
            wds[trial] = wd
        wd = np.mean(wds)
        std = np.std(wds)
        print(f"{i}: distance = {wd:.4f}, std = {std:.4f}")
        exit()

    print("\nWasserstein Results: gan 50 500")
    for i, pair in enumerate(gan_50_500_pairs):
        num_trials = 1
        wds = np.zeros(num_trials)
        for trial in range(num_trials):
            wd = compute_wasserstein_distance_drake(
                pair[0], pair[1], 1000
            )
            wds[trial] = wd
        wd = np.mean(wds)
        std = np.std(wds)
        print(f"{i}: distance = {wd:.4f}, std = {std:.4f}")
        
    print("\nWasserstein Results: no gan 10 2000")
    for i, pair in enumerate(no_gan_10_2000_pairs):
        num_trials = 1
        wds = np.zeros(num_trials)
        for trial in range(num_trials):
            wd = compute_wasserstein_distance_drake(
                pair[0], pair[1], 1000
            )
            wds[trial] = wd
        wd = np.mean(wds)
        std = np.std(wds)
        print(f"{i}: distance = {wd:.4f}, std = {std:.4f}")
    
    print("\nWasserstein Results: gan 10 2000")
    for i, pair in enumerate(gan_10_2000_pairs):
        num_trials = 1
        wds = np.zeros(num_trials)
        for trial in range(num_trials):
            wd = compute_wasserstein_distance_drake(
                pair[0], pair[1], 1000
            )
            wds[trial] = wd
        wd = np.mean(wds)
        std = np.std(wds)
        print(f"{i}: distance = {wd:.4f}, std = {std:.4f}")
    # sigma = 0.3
    # results = main(no_gan_10_2000_pairs_regular_emb, sigma=sigma)
    # print("MMD Results:")
    # for i, result in enumerate(results):
    #     print(f"MMD {i}: {result:.4f}")

    # sigma sweep
    # sigma = 0.1
    # for i in range(11):
    #     no_gan_50_500_results = main(no_gan_50_500_pairs, sigma)
    #     gan_50_500_results = main(gan_50_500_pairs, sigma)
    #     no_gan_10_2000_results = main(no_gan_10_2000_pairs, sigma)
    #     gan_10_2000_results = main(gan_10_2000_pairs, sigma)
        
    #     print("Sigma:", sigma)
    #     print("------------------------")
    #     print("50_500:", np.mean((no_gan_50_500_results - gan_50_500_results) / no_gan_50_500_results))
    #     print("50_500:", np.sum(no_gan_50_500_results > gan_50_500_results) / len(no_gan_50_500_results))
    #     print("10_2000:", np.mean((no_gan_10_2000_results - gan_10_2000_results) / no_gan_10_2000_results))
    #     print("10_2000:", np.sum(no_gan_10_2000_results > gan_10_2000_results) / len(no_gan_10_2000_results))
    #     print("------------------------")
    #     print("\n")

    #     sigma += 0.05