import time

import numpy as np
from tqdm import tqdm

from points.io import TrajectoryWriter
from points.sim.scenarios import RingedSystem


def main():
    """Run a sample simulation and write trajectory to Zarr file."""

    model = RingedSystem(G=1.0)
    writer = TrajectoryWriter("trajectory.zarr", n_particles=len(model.particles))

    for _ in tqdm(range(3000)):
        model.update(dt=0.01)
        positions = np.array([p.position for p in model.particles])

        try:
            writer.write_step(positions)
        except PermissionError:
            print("PermissionError: Retrying in 0.1 seconds...")
            time.sleep(0.1)
            writer.write_step(positions)


if __name__ == "__main__":
    main()
