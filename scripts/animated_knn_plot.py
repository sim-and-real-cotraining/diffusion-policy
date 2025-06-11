import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Patch, Rectangle, Circle, FancyArrowPatch
import os

class KNNAnimator:
    def __init__(self):
        self.data_dir = "experiments/knn/data"
        self.real_actions = None
        self.sim_actions = None
        self.percent_real_knn_real = None
        self.percent_real_knn_sim = None
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.cax = None  # Color bar axis
        self.axes = [self.ax1, self.ax2]
        # Animation parameters
        self.num_frames = None
        self.cmap = plt.cm.get_cmap('coolwarm')
        self.norm = plt.Normalize(vmin=0, vmax=1)
        
    def load_data(self):
        """Load the saved kNN data from experiments/knn/data directory."""
        # Load the saved numpy arrays
        self.real_actions = np.load(os.path.join(self.data_dir, "real_actions.npy"))
        self.real_actions = self.real_actions[1:-2]
        self.percent_real_knn_real = np.load(os.path.join(self.data_dir, "percent_real_knn_real.npy"))
        self.percent_real_knn_real = self.percent_real_knn_real[1:-2]
        
        self.sim_actions = np.load(os.path.join(self.data_dir, "sim_actions.npy"))
        self.percent_real_knn_sim = np.load(os.path.join(self.data_dir, "percent_real_knn_sim.npy"))
    
    def setup_plot(self):
        """Set up the figure and axes for the animation."""
        # Create figure with specific size and layout
        self.fig = plt.figure(figsize=(14, 8))
        
        # Create a GridSpec to manage the layout
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])
        
        # Create the two main axes
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        
        # Create the colorbar axis
        self.cax = self.fig.add_subplot(gs[0, 2])
        
        # Add the colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        cbar = plt.colorbar(sm, cax=self.cax)
        cbar.set_label(r"Percentage of kNN in $\mathcal{D}_R$", fontsize=16)
        
        # Increase horizontal spacing between subplots
        self.fig.subplots_adjust(wspace=0.3)
        
        # Set titles and labels
        self.ax1.set_title('Policy Rollout in Real', fontsize=18)
        self.ax2.set_title('Policy Rollout in Sim', fontsize=18)

        self.axes = [self.ax1, self.ax2]
        
        # Set axis labels
        for ax in [self.ax1, self.ax2]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            range_x = 0.4
            range_y = 0.45
            x_mid = 0.587
            y_mid = -0.0355
            ax.set_xlim([x_mid - range_x/2, x_mid + range_x/2])
            ax.set_ylim([y_mid - range_y/2, y_mid + range_y/2])

            # Set font sizes
            ax.set_xlabel('x', fontsize=16)
            ax.set_ylabel('y', fontsize=16)

            # Remove ticks
            ax.set_xticks([0.4, 0.5, 0.6, 0.7])
            ax.set_yticks([-0.2, -0.1, 0, 0.1])

            ax.set_aspect('equal', adjustable='box')
    
    def init(self):
        """Initialize the animation."""
        # Create patches for legend
        legend_target = Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.5)
        legend_circle = Circle((0, 0), radius=1, edgecolor='black', facecolor='none')

        for ax in self.axes:
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

        # Add legend to the first axis only
        self.ax1.legend([legend_target, legend_circle], 
                       ['Target T Pose', 'Initial Slider Pose'],
                       bbox_to_anchor=(1.1, -0.2),  # Moved up from -0.2
                       loc='lower center',
                       ncol=2,
                       fontsize=12)  # Added larger font size
        
        return []
    
    def animate(self, frame):
        """Update the animation for each frame."""
        num_real_frames = len(self.real_actions)
        num_sim_frames = len(self.sim_actions)
        total_animation_frames = num_real_frames + num_sim_frames
        
        # If we're in the extra frames at the end, just return empty list to keep frame static
        if frame > total_animation_frames:
            return []
            
        rollout = "REAL"
        if frame <= num_real_frames:
            # Update the real-world rollout
            ax = self.axes[0]
            actions = self.real_actions
            percent_real_knn = self.percent_real_knn_real
            num_arrows = frame
            rollout = "REAL"
        else:
            # Update the simulated rollout
            ax = self.axes[1]
            actions = self.sim_actions
            percent_real_knn = self.percent_real_knn_sim
            num_arrows = frame - num_real_frames
            rollout = "SIM"
        
        if num_arrows == 0:
            return []

        # Draw the trajectory
        start_idx = 2
        horizon = 8
        end_idx = start_idx + horizon

        cmap = plt.cm.get_cmap('coolwarm')
        norm = plt.Normalize(vmin=0, vmax=1)         

        # Only draw arrow for current trajectory
        arrow_idx = num_arrows - 1
        points = actions[arrow_idx]
        intensity = percent_real_knn[arrow_idx] 
        color = cmap(norm(intensity))
        
        if num_arrows == len(self.real_actions) and rollout == "REAL":
            x_coords = points[start_idx:end_idx-1, 0]
            y_coords = points[start_idx:end_idx-1, 1]
        else:
            x_coords = points[start_idx:end_idx, 0]
            y_coords = points[start_idx:end_idx, 1]
        ax.plot(x_coords, y_coords, color=color, alpha=1, linewidth=2, solid_capstyle='round')

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
                mutation_scale=15,  # Controls arrowhead size
            )
            ax.add_patch(arrow)  # Add the arrow to the plot

        return []
    
    def run(self):
        """Run the animation."""
        # Load the data
        self.load_data()
        
        # Set up the plot
        self.setup_plot()
        
        # Create the animation
        num_arrows = len(self.real_actions) + len(self.sim_actions)
        # Add 10 extra frames (2 seconds at 5 fps) for the final static frame
        fps = 5
        total_frames = num_arrows + 1 + 5 * fps
        
        anim = FuncAnimation(
            self.fig,
            self.animate,
            init_func=self.init,
            frames=total_frames,
            interval=1000 / fps,  # 200ms between frames (5 fps)
            blit=True
        )
        
        # Save animation
        writer = FFMpegWriter(fps=5)
        anim.save('plots/knn_animation.mp4', writer=writer)
        
        # Show the animation
        plt.show()

def main():
    animator = KNNAnimator()
    animator.run()

if __name__ == "__main__":
    main() 