from mayavi import mlab
import numpy as np
import argparse

colors = [
    (1, 0.5, 0.5),
    (0.5, 1, 0.5),
    (0.5, 0.5, 1),
]

parser = argparse.ArgumentParser()
parser.add_argument("--occ-file", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    occ_file = args.occ_file
    occ = np.load(occ_file)["occ"]
    mlab.figure(bgcolor=(1, 1, 1))  # Set background color to white

    for v in (0, 1, 2): # 0: unknown, 1: occupied, 2: free
        voxel_centers = np.argwhere(occ == v)
        x, y, z = voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2]

        point_cloud = mlab.points3d(
            x, y, z, mode="cube", color=colors[v], opacity=0.8, scale_factor=0.95
        )
    num_unknown = np.sum(occ == 0)
    num_occupied = np.sum(occ == 1)
    num_free = np.sum(occ == 2)
    num_total = num_unknown + num_occupied + num_free
    print(
        f"Unknown: {num_unknown/num_total}, Occupied: {num_occupied/num_total}, Free: {num_free/num_total}, Known: {(num_occupied+num_free)/num_total}    "
    )

    mlab.show()
