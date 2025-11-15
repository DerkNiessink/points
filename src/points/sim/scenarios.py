import numpy as np

from points.models.particles import ParticleSystem


class RingedSystem(ParticleSystem):
    """A particle system with a central star, a ringed planet, and rings."""

    def __init__(self, G=1.0):
        super().__init__(G)

        # === Central massive star (like a sun) ===
        sun_mass = 1000
        self.add_particle(
            mass=sun_mass, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)
        )

        # === Ringed Planet  ===
        planet_distance = 15
        planet_mass = 100

        # Calculate orbital velocity for the planet around the sun
        planet_orbital_velocity = np.sqrt(self.G * sun_mass / planet_distance)

        # Planet position and velocity
        planet_angle = 0.0  # Start at angle 0
        planet_pos_x = planet_distance * np.cos(planet_angle)
        planet_pos_y = planet_distance * np.sin(planet_angle)
        planet_vel_x = -planet_orbital_velocity * np.sin(planet_angle)
        planet_vel_y = planet_orbital_velocity * np.cos(planet_angle)

        # Add the ringed planet
        self.add_particle(
            mass=planet_mass,
            position=(planet_pos_x, planet_pos_y, 0.0),
            velocity=(planet_vel_x, planet_vel_y, 0.0),
        )

        # === Create rings around the planet ===
        # Multiple ring layers for a more realistic look
        ring_configs = [
            {"inner": 2, "outer": 2.1, "n": 1000, "color": (0, 0, 180)},  # Inner ring
            {
                "inner": 1.3,
                "outer": 1.4,
                "n": 1000,
                "color": (160, 0, 0),
            },  # Outer ring
        ]

        for ring in ring_configs:
            n_particles = ring["n"]
            inner_radius = ring["inner"]
            outer_radius = ring["outer"]

            for i in range(n_particles):
                # Distribute particles evenly around the ring
                angle = 2 * np.pi * i / n_particles
                # Random radius within ring bounds
                ring_radius = np.random.uniform(inner_radius, outer_radius)

                # Position relative to planet (in planet's frame)
                rel_x = ring_radius * np.cos(angle)
                rel_y = ring_radius * np.sin(angle)
                rel_z = 0  # Slight thickness

                # Convert to absolute position (add planet position)
                abs_x = planet_pos_x + rel_x
                abs_y = planet_pos_y + rel_y
                abs_z = rel_z

                # Calculate orbital velocity around the planet
                # Particles orbit the planet, but also move with the planet
                ring_orbital_velocity = np.sqrt(self.G * planet_mass / ring_radius)

                # Velocity in planet's frame (tangential to ring)
                rel_vel_x = -ring_orbital_velocity * np.sin(angle)
                rel_vel_y = ring_orbital_velocity * np.cos(angle)
                rel_vel_z = 0.0

                # Add planet's velocity to get absolute velocity
                abs_vel_x = planet_vel_x + rel_vel_x
                abs_vel_y = planet_vel_y + rel_vel_y
                abs_vel_z = rel_vel_z

                self.add_particle(
                    mass=0.0001,  # Small ring particles
                    position=(abs_x, abs_y, abs_z),
                    velocity=(abs_vel_x, abs_vel_y, abs_vel_z),
                )
