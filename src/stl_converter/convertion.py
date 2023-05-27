from pathlib import Path

import numpy as np
import voxelfuse as vf
from voxelfuse import VoxelModel


def save_numpy_as_mesh(voxel_model: VoxelModel, path_output: Path, show: bool = False):
    mesh = vf.Mesh.fromVoxelModel(voxel_model, (0.0, 1.0, 0.0, 1.0))
    mesh.export(str(path_output))
    if show:
        mesh.viewer(grids=True, name='mesh')


def load_and_process(filepath_in: Path, filepath_out: Path):
    data = np.load(str(filepath_in))
    data[data < 0.0] = 0.0
    data[data > 0.0] = 1.0
    model = VoxelModel(data, 1)
    save_numpy_as_mesh(model, filepath_out, True)


# Start Application
if __name__ == '__main__':
    file_in = Path("/media/kirrog/workdata/membransdata/newes/test/stpv/numpy/stpvM.npy")
    file_out = Path("modelResult.stl")
    load_and_process(file_in, file_out)
