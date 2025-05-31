import os
import argparse
import pickle
import torch
import shutil

from experiments.knn.knn import *
from experiments.knn.plotting import *

def visualize_knn(log_path, real_actions, sim_actions, real_embeddings, sim_embeddings, k, show_plots):
    # Set up paths and directories
    log_paths = []
    if os.path.isdir(log_path):
        log_paths = [os.path.join(log_path, f) for f in os.listdir(log_path) if f.endswith(".pkl")]
        log_paths.sort()
    else:
        log_paths = [log_path]

    for log_idx, log_path in enumerate(log_paths):
        print("Processing", log_path)
        with open(log_path, "rb") as f:
            data = pickle.load(f)

        # Extract the data and convert to torch tensors
        actions = torch.tensor(data['actions'])
        embeddings = torch.tensor(data['embeddings'])
        
        # Conduct knn experiment on actions
        action_dim = actions[0].flatten().shape[0]
        nearest_neighbors = trajectory_knn([sim_actions, real_actions], actions, k, sqrt_Q=torch.eye(action_dim))
     
        # Plot the results
        plot_knn_actions(
            actions[50:55].numpy(),
            nearest_neighbors,
            datasets=[sim_actions, real_actions],
            draw_target=True
        )

        # Conduct knn experiment on embeddings
        embedding_dim = embeddings[0].flatten().shape[0]
        nearest_neighbors = trajectory_knn([sim_embeddings, real_embeddings], embeddings, k, sqrt_Q=torch.eye(embedding_dim))
        
        # Plot the results
        plot_knn_actions(
            actions[50:55].numpy(),
            nearest_neighbors,
            datasets=[sim_actions, real_actions],
            draw_target=True
        )

        print()

if __name__ == "__main__":
    # TODO: add argparser
    log_path = "/home/adam/workspace/diffusion-planar-pushing/pickled_logs/50_500_3_1/real/0.pkl"
    real_actions_path = "data/experiments/real_actions.npy"
    sim_actions_path = "data/experiments/sim_actions.npy"
    real_embeddings_path = "data/experiments/real_embeddings.npy"
    sim_embeddings_path = "data/experiments/sim_embeddings.npy"

    real_actions = torch.tensor(np.load(real_actions_path))
    sim_actions = torch.tensor(np.load(sim_actions_path))
    real_embeddings = torch.tensor(np.load(real_embeddings_path))
    sim_embeddings = torch.tensor(np.load(sim_embeddings_path))
    k = 10
    show_plots = False

    visualize_knn(
        log_path=log_path,
        real_actions=real_actions,
        sim_actions=sim_actions,
        real_embeddings=real_embeddings,
        sim_embeddings=sim_embeddings,
        k=k,
        show_plots=show_plots
    )