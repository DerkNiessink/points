import time

from tqdm import tqdm

from points.io import TrajectoryWriter
from points.models.scenarios import GalaxyCollision, RingedSystem


def main():
    """Run a sample simulation and write trajectory to Zarr file."""

    model = GalaxyCollision(G=1.0)
    writer = TrajectoryWriter("trajectory.zarr", masses=model.masses)

    for _ in tqdm(range(1000)):
        model.update(dt=0.01)
        try:
            writer.write_step(model.positions, model.com)
        except PermissionError:
            print("PermissionError: Retrying in 0.1 seconds...")
            time.sleep(0.1)
            writer.write_step(model.positions, model.com)


if __name__ == "__main__":
    main()
