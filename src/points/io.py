import numpy as np
import zarr


class TrajectoryWriter:
    """Zarr writer for streaming simulation data of particle trajectories"""

    def __init__(self, filename: str, n_particles: int):
        self.root = zarr.open_group(filename, mode="w")
        self.n_particles = n_particles

        self.positions = self.root.create_array(
            "positions",
            shape=(0, n_particles, 3),
            dtype="float32",
        )

    def write_step(self, positions: np.ndarray):
        """Write a single time step of positions"""
        self.positions.append(positions[np.newaxis])


class TrajectoryReader:
    """Zarr reader for simulation data of particle trajectories"""

    def __init__(self, filename: str):
        self.root = zarr.open_group(filename, mode="r")
        self.positions = self.root["positions"]

    def positions_at_time(self, time_step: int) -> np.ndarray:
        """Return an (n_particles, 3) array of positions at the given time step."""
        return np.asarray(self.positions[time_step])

    def time_steps(self) -> int:
        """Return the total number of time steps stored."""
        return self.positions.shape[0]

    def n_particles(self) -> int:
        """Return the number of particles stored."""
        return self.positions.shape[1]
