"""Model layer: Physics and data logic for particles."""

import numpy as np

from points.models.fast_forces import update_positions_rk4


class Particle:
    """A particle with mass, position, and velocity in 3D space."""

    def __init__(
        self,
        mass: float,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.mass = mass
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)


class ParticleSystem:
    """Manages a collection of particles with gravitational interactions."""

    def __init__(self, gravitational_constant: float = 1.0):
        self.particles: list[Particle] = []
        self.G = gravitational_constant

        self._positions: np.ndarray | None = None
        self._velocities: np.ndarray | None = None
        self._masses: np.ndarray | None = None

    def add_particle(
        self,
        mass: float,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Particle:
        """Add a new particle to the system."""
        particle = Particle(mass, position, velocity)
        self.particles.append(particle)
        return particle

    def _build_arrays(self):
        """Build numpy arrays from particle list."""
        n = len(self.particles)
        self._positions = np.zeros((n, 3), dtype=np.float32)
        self._velocities = np.zeros((n, 3), dtype=np.float32)
        self._masses = np.zeros(n, dtype=np.float32)

        for i, p in enumerate(self.particles):
            self._positions[i] = p.position
            self._velocities[i] = p.velocity
            self._masses[i] = p.mass

    def _update_particles(self):
        """Write array data back to particle objects."""
        for i, p in enumerate(self.particles):
            p.position = self._positions[i].copy()
            p.velocity = self._velocities[i].copy()

    def update(self, dt: float, softening: float = 0.01):
        """Update all particles using RK4 integration.

        Args:
            dt: Time step for physics simulation
            softening: Prevents singularities when particles are very close
        """
        if not self.particles:
            return

        # Build arrays once on first update
        if self._positions is None:
            self._build_arrays()

        # Run RK4 integration (modifies arrays in-place)
        update_positions_rk4(
            self._positions,
            self._velocities,
            self._masses,
            self.G,
            softening,
            dt,
        )

        # Sync back to particles
        self._update_particles()
