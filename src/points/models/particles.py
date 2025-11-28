import numpy as np

from points.models.fast_forces import update_center_of_mass, update_positions_rk4


class Particles:
    """Manages a collection of particles with gravitational interactions."""

    def __init__(self, gravitational_constant: float = 1.0):
        self.G = gravitational_constant

        self.positions = np.empty((0, 3), dtype=np.float32)
        self.velocities = np.empty((0, 3), dtype=np.float32)
        self.masses = np.empty(0, dtype=np.float32)
        self.com = np.zeros(3, dtype=np.float32)

    def add_particle(
        self,
        mass: float,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Add a new particle to the system."""
        self.positions = np.vstack([self.positions, position])
        self.velocities = np.vstack([self.velocities, velocity])
        self.masses = np.append(self.masses, mass)

    def update(self, dt: float, softening: float = 0.01):
        """Update all particles using RK4 integration."""
        update_positions_rk4(
            self.positions,
            self.velocities,
            self.masses,
            self.G,
            softening,
            dt,
        )
        update_center_of_mass(self.com, self.positions, self.masses)
