import time

from tqdm import tqdm

from points.io import TrajectoryWriter
from points.sim.scenarios import RingedSystem


def main():
    """Run a sample simulation and write trajectory to Zarr file."""

    model = RingedSystem(G=1.0)
    writer = TrajectoryWriter("trajectory.zarr", masses=model.masses)

    for _ in tqdm(range(500)):
        model.update(dt=0.01)
        try:
            writer.write_step(model.positions)
        except PermissionError:
            print("PermissionError: Retrying in 0.1 seconds...")
            time.sleep(0.1)
            writer.write_step(model.positions)


if __name__ == "__main__":
    main()
