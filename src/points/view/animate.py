import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

from points.io import TrajectoryReader


def animate_trajectories(
    filename: str,
    save_as: str,
    interval: int = 50,
    figsize: tuple = (10, 10),
):
    """Create an animated visualization of particle trajectories.

    Args:
        filename: Path to the zarr file with trajectory data
        save_as: Filename to save animation (e.g. 'output.gif')
        interval: Milliseconds between frames
        figsize: Figure size (width, height)
    """

    reader = TrajectoryReader(filename)
    n_steps = reader.time_steps()
    n_particles = reader.n_particles()

    print(f"Loaded {n_particles} particles with {n_steps} time steps")

    # Calculate marker sizes from masses
    # Using square root to dampen the huge range of masses
    masses = reader.masses()
    marker_sizes = 2 + np.sqrt(masses) * 5

    # Set up the figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Hide axes and set background to black
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()

    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(-3, 3)

    # Initialize scatter plot with first frame
    initial_positions = reader.positions_at_time(0)
    scatter = ax.scatter(
        initial_positions[:, 0],
        initial_positions[:, 1],
        initial_positions[:, 2],
        s=marker_sizes,
        alpha=0.8,
    )

    def update(frame):
        """Update animation frame."""
        positions = reader.positions_at_time(frame)
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        return (scatter,)

    # Create animation
    anim = FuncAnimation(fig, update, frames=n_steps, interval=interval, blit=False)

    # Save animation
    print(f"\nSaving animation to {save_as}...")
    save_pbar = tqdm(total=n_steps, desc="Saving frames")

    def progress_callback(current_frame, total_frames):
        save_pbar.n = current_frame
        save_pbar.refresh()

    fps = 1000 // interval
    writer = PillowWriter(fps=fps)
    anim.save(save_as, writer=writer, progress_callback=progress_callback, dpi=80)
    save_pbar.close()
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    animate_trajectories(
        filename="trajectory.zarr",
        save_as="orbit.gif",
        interval=70,
    )
