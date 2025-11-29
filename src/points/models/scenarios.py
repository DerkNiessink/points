import numpy as np

from points.models.particles import Particles


class RingedSystem(Particles):
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
            {"inner": 1.8, "outer": 1.9, "n": 3000},
            {"inner": 1, "outer": 1.2, "n": 1000},
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
                rel_z = 0.0  # Flat ring in planet's equatorial plane

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
                    mass=0.001,  # Small ring particles
                    position=(abs_x, abs_y, abs_z),
                    velocity=(abs_vel_x, abs_vel_y, abs_vel_z),
                )


class GalaxyCollision(Particles):
    """Two ringed galaxies moving towards each other for a collision."""

    def __init__(self, G=1.0):
        super().__init__(G)

        # Galaxy parameters
        center_mass = 1000
        ring_radius = 2.5
        ring_thickness = 0.2
        n_ring_particles = 10000
        ring_thickness_z = 0.1

        # Collision parameters
        separation = 25
        y_offset = 8
        velocity = 9

        # Create both galaxies
        galaxy_configs = [
            (
                np.array([-separation / 2, y_offset / 2, 0.0]),
                np.array([velocity, 0.0, 0.0]),
            ),
            (
                np.array([separation / 2, -y_offset / 2, 0.0]),
                np.array([-velocity, 0.0, 0.0]),
            ),
        ]

        for center_pos, center_vel in galaxy_configs:
            # Add central massive particle
            self.add_particle(
                mass=center_mass,
                position=tuple(center_pos),
                velocity=tuple(center_vel),
            )

            # Add ring particles
            for i in range(n_ring_particles):
                angle = 2 * np.pi * i / n_ring_particles

                # Random radius for thickness in xy-plane
                r = ring_radius + np.random.uniform(-ring_thickness, ring_thickness)

                # Position in ring with thickness in xy and z
                rel_pos = np.array(
                    [
                        r * np.cos(angle),
                        r * np.sin(angle),
                        np.random.uniform(-ring_thickness_z, ring_thickness_z),
                    ]
                )

                # Orbital velocity (using ring_radius for stable orbit)
                v_orbital = np.sqrt(self.G * center_mass / ring_radius)
                rel_vel = np.array(
                    [
                        -v_orbital * np.sin(angle),
                        v_orbital * np.cos(angle),
                        0.0,
                    ]
                )

                # Convert to absolute coordinates
                abs_pos = center_pos + rel_pos
                abs_vel = center_vel + rel_vel

                self.add_particle(
                    mass=0.01,
                    position=tuple(abs_pos),
                    velocity=tuple(abs_vel),
                )
