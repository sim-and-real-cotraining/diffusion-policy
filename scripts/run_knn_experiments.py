import os
import argparse
import pickle
import torch
import shutil

from experiments.knn.knn import *
from experiments.knn.plotting import *

def knn_experiment(log_path, output_dir, real_actions, sim_actions, real_embeddings, sim_embeddings, k, show_plots):
    # Set up paths and directories
    log_paths = []
    if os.path.isdir(log_path):
        log_paths = [os.path.join(log_path, f) for f in os.listdir(log_path) if f.endswith(".pkl")]
        log_paths.sort()
    else:
        log_paths = [log_path]
    
    if os.path.exists(output_dir):
        resp = input(f"Output directory {output_dir} already exists. Overwrite? (y/n): ")
        if resp.lower() != 'y':
            return
        else:
            resp = input(f"Are you sure you want to delete {output_dir}? (y/n): ")
            if resp.lower() == 'y':
                shutil.rmtree(output_dir)
            else:
                return
    os.makedirs(output_dir, exist_ok=True)
    
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
        dataset_index_matrix = neighbors_to_dataset_indices(nearest_neighbors)
        percent_real_knn = neighbors_to_dataset_intensity(nearest_neighbors)

        # Plot the results
        # plot_integer_grid(
        #     dataset_index_matrix, 
        #     colors=['blue', 'red'], 
        #     legend=['Sim', 'Real'], 
        #     title='kNN on Actions', 
        #     x_label='Time Step', 
        #     y_label='Neighbor Index', 
        #     save_path=os.path.join(output_dir, f"actions_knn_grid_{log_idx}.pdf"), # TODO: arguments
        #     show_plot=show_plots
        # )

        plot_points_with_intensity(
            actions.numpy()[1:-2],
            percent_real_knn[1:-2],
            title='Real-World Rollout',
            save_path=os.path.join(output_dir, f"actions_knn_intensity_{log_idx}.pdf"),
            show_plot=show_plots,
            show_colorbar = True
        )

        # plot_points_with_intensity_iros(
        #     actions.numpy()[1:-2],
        #     percent_real_knn[1:-2],
        #     actions.numpy()[1:-2],
        #     percent_real_knn[1:-2],
        #     title_1='Real-World Rollout',
        #     title_2='Simulated Rollout',
        #     cmap_label=r"Percentage of kNN in $\mathcal{D}_R$",
        #     save_path=os.path.join(output_dir, f"knn.pdf"),
        #     show_plot=show_plots,
        # )

        # # Conduct knn experiment on embeddings
        # embedding_dim = embeddings[0].flatten().shape[0]
        # nearest_neighbors = trajectory_knn([sim_embeddings, real_embeddings], embeddings, k, sqrt_Q=torch.eye(embedding_dim))
        # dataset_index_matrix = neighbors_to_dataset_indices(nearest_neighbors)
        # percent_real_knn = neighbors_to_dataset_intensity(nearest_neighbors)

        # # Plot the results
        # plot_integer_grid(
        #     dataset_index_matrix, 
        #     colors=['blue', 'red'], 
        #     legend=['Sim', 'Real'], 
        #     title='kNN on Embeddings', 
        #     x_label='Time Step', 
        #     y_label='Neighbor Index', 
        #     save_path=os.path.join(output_dir, f"embeddings_knn_grid_{log_idx}.pdf"), # TODO: arguments
        #     show_plot=show_plots
        # )

        # plot_points_with_intensity(
        #     actions.numpy(),
        #     percent_real_knn,
        #     title='kNN on Embeddings',
        #     save_path=os.path.join(output_dir, f"embeddings_knn_intensity_{log_idx}.pdf"),
        #     show_plot=show_plots
        # )

if __name__ == "__main__":
    # TODO: add argparser
    run = "real_2"
    log_path = f"/home/adam/workspace/diffusion-planar-pushing/pickled_logs/50_500_3_1/iros_{run}"
    output_dir = f"plots/iros_knn/50_500_3_1/{run}"
    real_actions_path = "data/experiments/real_actions.npy"
    sim_actions_path = "data/experiments/sim_actions.npy"
    real_embeddings_path = "data/experiments/real_embeddings.npy"
    sim_embeddings_path = "data/experiments/sim_embeddings.npy"

    real_actions = torch.tensor(np.load(real_actions_path))
    sim_actions = torch.tensor(np.load(sim_actions_path))
    real_embeddings = torch.tensor(np.load(real_embeddings_path))
    sim_embeddings = torch.tensor(np.load(sim_embeddings_path))
    k = 3
    show_plots = True

    knn_experiment(
        log_path=log_path,
        output_dir=output_dir,
        real_actions=real_actions,
        sim_actions=sim_actions,
        real_embeddings=real_embeddings,
        sim_embeddings=sim_embeddings,
        k=k,
        show_plots=show_plots
    )