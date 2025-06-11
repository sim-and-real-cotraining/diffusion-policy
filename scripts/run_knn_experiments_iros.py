import os
import argparse
import pickle
import torch
import shutil

from experiments.knn.knn import *
from experiments.knn.plotting import *

def knn_experiment(log_path_real, log_path_sim, output_dir, real_actions, sim_actions, real_embeddings, sim_embeddings, k, show_plots):
    # delete and recreate output dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing", log_path_real)
    with open(log_path_real, "rb") as f:
        data = pickle.load(f)

    # Extract the data and convert to torch tensors
    actions_real = torch.tensor(data['actions'])
    embeddings_real = torch.tensor(data['embeddings'])
    action_dim = actions_real[0].flatten().shape[0]
    embedding_dim = embeddings_real[0].flatten().shape[0]

    # Concatenate actions and embeddings
    o_a_real_rollout = torch.concat([embeddings_real.reshape(-1, embedding_dim), actions_real.reshape(-1, action_dim)], dim=1)
    o_a_real_dataset = torch.concat(
        [real_embeddings.reshape(-1, embedding_dim), real_actions.reshape(-1, action_dim)], 
        dim=1
    )
    o_a_sim_dataset = torch.concat(
        [sim_embeddings.reshape(-1, embedding_dim), sim_actions.reshape(-1, action_dim)], 
        dim=1
    )

    # Construct the Q matrix
    tau = 0.0 # Note that action_dim < embedding_dim, hence small tau
    Q = torch.eye(embedding_dim + action_dim)
    Q[:embedding_dim, :embedding_dim] = torch.eye(embedding_dim) * tau

    # Conduct knn experiment on actions
    nearest_neighbors_real = trajectory_knn([o_a_sim_dataset, o_a_real_dataset], o_a_real_rollout, k, sqrt_Q=Q)
    percent_real_knn_real = neighbors_to_dataset_intensity(nearest_neighbors_real)
    
    print("Processing", log_path_sim)
    with open(log_path_sim, "rb") as f:
        data = pickle.load(f)

    # Extract the data and convert to torch tensors
    actions_sim = torch.tensor(data['actions'])
    embeddings_sim = torch.tensor(data['embeddings'])

    # Concatenate actions and embeddings
    o_a_sim_rollout = torch.concat([embeddings_sim.reshape(-1, embedding_dim), actions_sim.reshape(-1, action_dim)], dim=1)
    
    # Conduct knn experiment on actions
    nearest_neighbors_sim = trajectory_knn([o_a_sim_dataset, o_a_real_dataset], o_a_sim_rollout, k, sqrt_Q=Q)
    percent_real_knn_sim = neighbors_to_dataset_intensity(nearest_neighbors_sim)    
    # print(percent_real_knn_sim)

    # Save data as numpy arrays
    save_path = "experiments/knn/data"
    np.save(os.path.join(save_path, "real_actions.npy"), actions_real.numpy())
    np.save(os.path.join(save_path, "sim_actions.npy"), actions_sim.numpy())
    np.save(os.path.join(save_path, "percent_real_knn_real.npy"), percent_real_knn_real)
    np.save(os.path.join(save_path, "percent_real_knn_sim.npy"), percent_real_knn_sim)

    breakpoint()
    plot_points_with_intensity_iros(
        actions_real.numpy()[1:-2],
        percent_real_knn_real[1:-2],
        actions_sim.numpy(),
        percent_real_knn_sim,
        title_1='Real-World Rollout',
        title_2='Simulated Rollout',
        cmap_label=r"Percentage of kNN in $\mathcal{D}_R$",
        figsize=(12, 6),
        save_path=os.path.join(output_dir, f"knn.pdf"),
        show_plot=show_plots,
    )

if __name__ == "__main__":
    # TODO: add argparser
    log_path_real = f"/home/adam/workspace/diffusion-planar-pushing/pickled_logs/50_500_3_1/iros_real_2/2.pkl"
    log_path_sim = f"/home/adam/workspace/diffusion-planar-pushing/pickled_logs/50_500_3_1/iros_sim_2/2.pkl"
    output_dir = f"plots/iros_knn/50_500_3_1/sim_and_real"
    real_actions_path = "/home/adam/workspace/gcs-diffusion/data/experiments/real_actions.npy"
    sim_actions_path = "/home/adam/workspace/gcs-diffusion/data/experiments/sim_actions.npy"
    real_embeddings_path = "/home/adam/workspace/gcs-diffusion/data/experiments/real_embeddings.npy"
    sim_embeddings_path = "/home/adam/workspace/gcs-diffusion/data/experiments/sim_embeddings.npy"

    real_actions = torch.tensor(np.load(real_actions_path))
    sim_actions = torch.tensor(np.load(sim_actions_path))
    real_embeddings = torch.tensor(np.load(real_embeddings_path))
    sim_embeddings = torch.tensor(np.load(sim_embeddings_path))
    k = 3
    show_plots = True

    knn_experiment(
        log_path_real=log_path_real,
        log_path_sim=log_path_sim,
        output_dir=output_dir,
        real_actions=real_actions,
        sim_actions=sim_actions,
        real_embeddings=real_embeddings,
        sim_embeddings=sim_embeddings,
        k=k,
        show_plots=show_plots
    )