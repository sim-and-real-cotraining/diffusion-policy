import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
from typing import Union

def plot_integer_grid(
    integer_matrix: np.ndarray,  # A 2D numpy array of integers >= 0
    colors: list=['blue', 'red'],
    legend: list=['Sim', 'Real'],
    title: str = '',
    x_label: str = '',
    y_label: str = '',
    save_path: str='',
    show_plot: bool=False,
):   
    k, n = integer_matrix.shape
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots()
    integer_matrix = integer_matrix.astype(np.int32)
    cax = ax.imshow(integer_matrix, aspect='auto', cmap=cmap)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    
    # Set x-axis labels every 10 columns
    x_ticks = np.arange(-1, n-1, 10)  # Every 10 columns
    ax.set_xticks(x_ticks)  # Positions of the x ticks
    ax.set_xticklabels(x_ticks + 1)  # Labels for x ticks (starting from 1)

    # Set y-axis labels (starting from 1)
    ax.set_yticks(np.arange(k))  # Positions of the y ticks
    ax.set_yticklabels(np.arange(1, k + 1))  # Labels for y ticks (starting from 1)
    
    # Add titles to the plot
    if title != '':
        ax.set_title(title)
    if x_label != '':
        ax.set_xlabel(x_label)
    if y_label != '':
        ax.set_ylabel(y_label)
    
    # Add a legend
    legend_elements = [Patch(color=colors[i], label=legend[i]) for i in range(len(colors))]
    ax.legend(handles=legend_elements, loc='upper right')

    # Display/save the plot
    if save_path != '':
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

from matplotlib.ticker import MaxNLocator

def plot_points_with_intensity(
    points_array: np.ndarray, 
    intensity_array: np.ndarray, 
    title: str ='',
    x_label: str ='x',
    y_label: str ='y',
    cmap_label: str = "Percentage of kNN in Real",
    draw_target: bool=True,
    save_path: str='',
    show_plot: bool=False,
    show_colorbar: bool=True,  # Option to display the colorbar
):
    """
    Plots trajectories as thin lines with a constant alpha, ensuring the arrow is exactly at the end.
    Font sizes adjusted for better readability.
    """
    assert points_array.shape[0] == len(intensity_array), (
        "points_array.shape[0] must match the length of intensity_array."
    )
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    if draw_target:
        # Draw the target T object
        top_rect = Rectangle((0.50445, 0), 0.1651, 0.04064, 
                             linewidth=1, facecolor='lightgreen', alpha=0.5)
        ax.add_patch(top_rect)
        bot_rect = Rectangle((0.56668, -0.12192), 0.04064, 0.12192, 
                             linewidth=1, facecolor='lightgreen', alpha=0.5)
        ax.add_patch(bot_rect)

        # Draw the pusher home position (circle)
        circle_center = (0.587, 0.15)
        circle_radius = 0.015
        circle = Circle(circle_center, circle_radius, edgecolor='black', 
                        facecolor='none', linewidth=1)
        ax.add_patch(circle)

    # Create a colormap that transitions from blue to red
    cmap = plt.cm.get_cmap('coolwarm')

    # Define the portion of each trajectory to plot
    start_idx = 2
    horizon = 8
    end_idx = start_idx + horizon
    
    # Plot each trajectory
    for i in range(points_array.shape[0]):
        points = points_array[i]
        intensity = intensity_array[i]
        
        # Map intensity to a color in the colormap
        color = cmap(intensity)

        # Extract the relevant segment of the trajectory
        x_coords = points[start_idx:end_idx, 0]
        y_coords = points[start_idx:end_idx, 1]

        # Draw a thin constant alpha line
        ax.plot(x_coords, y_coords, color=color, alpha=1, linewidth=1.5, solid_capstyle='round')

        # Ensure the arrow is at the exact end of the trajectory
        if len(x_coords) > 1:  # Ensure at least two points exist
            end_coord_1 = (x_coords[-1], y_coords[-1])
            end_coord_2 = (
                x_coords[-1] + 0.75*(x_coords[-1] - x_coords[-2]), 
                y_coords[-1] + 0.75*(y_coords[-1] - y_coords[-2])
            )
            arrow = FancyArrowPatch(
                end_coord_1,  # Start at second-to-last point
                end_coord_2,  # End exactly at the last point
                arrowstyle="->",  # Standard arrow
                color=color,
                linewidth=2,
                mutation_scale=10,  # Controls arrowhead size
            )
            ax.add_patch(arrow)  # Add the arrow to the plot

    # Force axis limits to be applied
    ax.autoscale(False)  

    # Explicitly set axis limits
    range_x = 0.4
    range_y = 0.45
    x_mid = 0.587
    y_mid = -0.0355
    ax.set_xlim([x_mid - range_x/2, x_mid + range_x/2])
    ax.set_ylim([y_mid - range_y/2, y_mid + range_y/2])

    # Set font sizes
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    ax.set_title(title, fontsize=20)

    # Adjust tick sizes
    # ax.tick_params(axis='both', labelsize=14)
    # Remove all ticks from both axes
    ax.set_xticks([0.4, 0.5, 0.6, 0.7])
    ax.set_yticks([-0.2, -0.1, 0, 0.1])


    # Add colorbar if enabled
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, label=cmap_label)
        cbar.ax.tick_params(labelsize=14)  # Set colorbar tick size
        cbar.set_label(cmap_label, fontsize=20)

    ax.set_aspect('equal', adjustable='box')

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

def plot_points_with_intensity_iros(
    points_array_1: np.ndarray, 
    intensity_array_1: np.ndarray, 
    points_array_2: np.ndarray, 
    intensity_array_2: np.ndarray, 
    title_1: str ='Plot 1',
    title_2: str ='Plot 2',
    x_label: str ='x',
    y_label: str ='y',
    cmap_label: str = "Percentage of kNN in Real",
    draw_target: bool=True,
    save_path: str='',
    figsize=(10, 6),
    show_plot: bool=False
):
    """
    Plots two trajectory plots side by side while sharing a single colorbar.
    """
    assert points_array_1.shape[0] == len(intensity_array_1), (
        "points_array_1.shape[0] must match the length of intensity_array_1."
    )
    assert points_array_2.shape[0] == len(intensity_array_2), (
        "points_array_2.shape[0] must match the length of intensity_array_2."
    )

    # Create a colormap that transitions from blue to red
    cmap = plt.cm.get_cmap('coolwarm')

    # Create a shared colorbar scale
    norm = plt.Normalize(vmin=min(intensity_array_1.min(), intensity_array_2.min()),
                         vmax=max(intensity_array_1.max(), intensity_array_2.max()))

    # Set up figure and axes for two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.1)  # Adjust as needed (0.05 for even tighter spacing)

    def plot_trajectories(ax, points_array, intensity_array, title, real=True):
        """ Helper function to plot trajectories on an axis. """
        if draw_target:
            # Draw the target T object
            top_rect = Rectangle((0.50445, 0), 0.1651, 0.04064, 
                                 linewidth=1, facecolor='lightgreen', alpha=0.5)
            ax.add_patch(top_rect)
            bot_rect = Rectangle((0.56668, -0.12192), 0.04064, 0.12192, 
                                 linewidth=1, facecolor='lightgreen', alpha=0.5)
            ax.add_patch(bot_rect)

            # Draw the pusher home position (circle)
            circle_center = (0.587, 0.15)
            circle_radius = 0.015
            circle = Circle(circle_center, circle_radius, edgecolor='black', 
                            facecolor='none', linewidth=1)
            ax.add_patch(circle)

        # Define the portion of each trajectory to plot
        start_idx = 2
        horizon = 8
        end_idx = start_idx + horizon

        # Plot each trajectory
        for i in range(points_array.shape[0]):
            points = points_array[i]
            intensity = intensity_array[i]
            
            # Map intensity to a color in the colormap
            color = cmap(norm(intensity))

            # Extract the relevant segment of the trajectory
            if i == len(points_array) - 1 and real:
                x_coords = points[start_idx:end_idx-1, 0]
                y_coords = points[start_idx:end_idx-1, 1]
            else:
                x_coords = points[start_idx:end_idx, 0]
                y_coords = points[start_idx:end_idx, 1]

            # Draw a thin constant alpha line
            ax.plot(x_coords, y_coords, color=color, alpha=1, linewidth=2, solid_capstyle='round')

            # Ensure the arrow is at the exact end of the trajectory
            if len(x_coords) > 1:  # Ensure at least two points exist
                end_coord_1 = (x_coords[-1], y_coords[-1])
                end_coord_2 = (
                    x_coords[-1] + 0.75*(x_coords[-1] - x_coords[-2]), 
                    y_coords[-1] + 0.75*(y_coords[-1] - y_coords[-2])
                )
                # end_coord_1 = (x_coords[-2], y_coords[-2])
                # end_coord_2 = (x_coords[-1], y_coords[-1])
                arrow = FancyArrowPatch(
                    end_coord_1,  # Start at second-to-last point
                    end_coord_2,  # End exactly at the last point
                    arrowstyle="->",  # Standard arrow
                    color=color,
                    linewidth=2,
                    mutation_scale=15,  # Controls arrowhead size
                )
                ax.add_patch(arrow)  # Add the arrow to the plot

        # Explicitly set axis limits
        range_x = 0.4
        range_y = 0.45
        x_mid = 0.587
        y_mid = -0.0355
        ax.set_xlim([x_mid - range_x/2, x_mid + range_x/2])
        ax.set_ylim([y_mid - range_y/2, y_mid + range_y/2])

        # Set font sizes
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_title(title, fontsize=18)

        # Remove ticks
        ax.set_xticks([0.4, 0.5, 0.6, 0.7])
        ax.set_yticks([-0.2, -0.1, 0, 0.1])

        ax.set_aspect('equal', adjustable='box')

    # Plot both subplots
    plot_trajectories(axes[0], points_array_1, intensity_array_1, title_1, real=True)
    plot_trajectories(axes[1], points_array_2, intensity_array_2, title_2, real=False)

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position colorbar outside
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, label=cmap_label)
    cbar.ax.tick_params(labelsize=14)  # Set colorbar tick size
    cbar.set_label(cmap_label, fontsize=18)

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()



def plot_knn_actions(
    actions: np.ndarray,
    nearest_neighbors: list,
    datasets: list,
    draw_target: bool=True
):
    for action_index, action in enumerate(actions):
        for rank, neighbor in enumerate(nearest_neighbors[action_index]):
            plt.figure(figsize=(10, 8))

            if draw_target:
                # Draw the target T position
                top_rect= Rectangle((0.50445, 0), 0.1651, 0.04064, linewidth=1, facecolor='lightgreen', alpha=0.5)
                plt.gca().add_patch(top_rect)
                bot_rect= Rectangle((0.56668, -0.12192), 0.04064, 0.12192, linewidth=1, facecolor='lightgreen', alpha=0.5)
                plt.gca().add_patch(bot_rect)

                # Draw the pusher home position
                circle_center = (0.587, 0.15)
                circle_radius = 0.015
                circle = Circle(circle_center, circle_radius, edgecolor='black', facecolor='none', linewidth=1)
                plt.gca().add_patch(circle)

            # plot action (16x2)
            plt.scatter(action[:, 0], action[:, 1], color='black', s=10)
            action_neighbor = datasets[neighbor.dataset_index][neighbor.index].reshape(action.shape)
            color = 'blue' if neighbor.dataset_index == 0 else 'red'
            plt.scatter(action_neighbor[:, 0], action_neighbor[:, 1], color=color, s=10)
            plt.title(f"Action {action_index} Neighbor {rank+1} Distance {neighbor.distance:.6f}")

            # Plot x and y ranges
            if draw_target:
                range_x = 0.5
                range_y = 0.6
                x_mid = 0.587
                y_mid = -0.0355
                plt.xlim([x_mid - range_x/2, x_mid + range_x/2])
                plt.ylim([y_mid - range_y/2, y_mid + range_y/2])

            plt.show()
            plt.close()