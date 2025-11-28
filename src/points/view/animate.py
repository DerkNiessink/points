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
    """Create an animated visualization of particle trajectories."""

    reader = TrajectoryReader(filename)
    print(
        f"Loaded {reader.n_particles()} particles with {reader.time_steps()} time steps"
    )

    # Calculate marker sizes from masses using square root to dampen range
    marker_sizes = 2 + np.sqrt(reader.masses()) * 3

    # Set up figure with black background and hidden axes
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()
    ax.set(xlim=(-12, 12), ylim=(-12, 12), zlim=(-3, 3))

    # Initialize scatter plot
    pos = reader.positions_at_time(0)
    scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=marker_sizes, alpha=0.8)

    # Get center of mass trajectory for re-centering
    com_data = reader.centers_of_mass()

    def update(frame):
        pos = reader.positions_at_time(frame)

        # Shift positions so center of mass is at origin
        # Handle edge case where frame might equal n_steps
        com_frame = min(frame, len(com_data) - 1)
        pos_centered = pos - com_data[com_frame]
        scatter._offsets3d = (
            pos_centered[:, 0],
            pos_centered[:, 1],
            pos_centered[:, 2],
        )

        return (scatter,)

    # Create and save animation with progress bar
    anim = FuncAnimation(
        fig, update, frames=reader.time_steps(), interval=interval, blit=False
    )

    print(f"\nSaving animation to {save_as}...")
    with tqdm(total=reader.time_steps(), desc="Saving frames") as pbar:
        writer = PillowWriter(fps=1000 // interval)
        anim.save(
            save_as,
            writer=writer,
            progress_callback=lambda i, n: pbar.update(1) if i > pbar.n else None,
            dpi=80,
        )

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    animate_trajectories(
        filename="trajectory.zarr",
        save_as="orbit.gif",
        interval=70,
    )
