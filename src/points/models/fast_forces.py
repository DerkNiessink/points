import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True, parallel=True)
def _calculate_accelerations_numba(
    positions: np.ndarray,
    masses: np.ndarray,
    G: float,
    softening: float,
) -> np.ndarray:
    """Calculate accelerations for all particles.

    Returns acceleration (force per unit mass) to avoid redundant division.
    """
    n = positions.shape[0]
    accelerations = np.zeros((n, 3))
    softening_sq = softening * softening
    masses_times_G = masses * G

    for i in prange(n):
        acc = np.zeros(3)
        ix, iy, iz = positions[i]
        for j in range(n):
            if i != j:
                # Vector from i to j
                dx = positions[j, 0] - ix
                dy = positions[j, 1] - iy
                dz = positions[j, 2] - iz

                # Distance squared with softening
                r_sq = dx * dx + dy * dy + dz * dz + softening_sq

                # Acceleration magnitude: G * m_j / r²
                # Direction: r_vec / r
                # Combined: G * m_j * r_vec / r³
                acc_mag = masses_times_G[j] / r_sq**1.5

                acc[0] += acc_mag * dx
                acc[1] += acc_mag * dy
                acc[2] += acc_mag * dz

        accelerations[i] = acc

    return accelerations


@jit(nopython=True, fastmath=True)
def update_center_of_mass(
    com: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
) -> None:
    """Update center of mass vector in place."""
    total_mass = np.sum(masses)
    com[:] = 0
    for i in range(len(masses)):
        com[0] += masses[i] * positions[i, 0]
        com[1] += masses[i] * positions[i, 1]
        com[2] += masses[i] * positions[i, 2]
    com /= total_mass


@jit(nopython=True, fastmath=True)
def update_positions_rk4(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    G: float,
    softening: float,
    dt: float,
):
    """Update positions and velocities using RK4 integration."""
    n = positions.shape[0]
    dt_half = dt * 0.5
    dt_sixth = dt / 6.0

    # Store original state
    pos0 = positions.copy()
    vel0 = velocities.copy()

    # k1: acceleration at current state
    k1_acc = _calculate_accelerations_numba(pos0, masses, G, softening)
    k1_vel = vel0.copy()

    # k2: acceleration at midpoint using k1
    pos_k2 = pos0 + k1_vel * dt_half
    vel_k2 = vel0 + k1_acc * dt_half
    k2_acc = _calculate_accelerations_numba(pos_k2, masses, G, softening)

    # k3: acceleration at midpoint using k2
    pos_k3 = pos0 + vel_k2 * dt_half
    vel_k3 = vel0 + k2_acc * dt_half
    k3_acc = _calculate_accelerations_numba(pos_k3, masses, G, softening)

    # k4: acceleration at endpoint using k3
    pos_k4 = pos0 + vel_k3 * dt
    vel_k4 = vel0 + k3_acc * dt
    k4_acc = _calculate_accelerations_numba(pos_k4, masses, G, softening)

    # Update with weighted average
    for i in range(n):
        # Velocity update: integrate acceleration
        velocities[i] = vel0[i] + dt_sixth * (
            k1_acc[i] + 2.0 * k2_acc[i] + 2.0 * k3_acc[i] + k4_acc[i]
        )

        # Position update: integrate velocity
        positions[i] = pos0[i] + dt_sixth * (
            k1_vel[i] + 2.0 * vel_k2[i] + 2.0 * vel_k3[i] + vel_k4[i]
        )
