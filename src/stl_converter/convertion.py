from pathlib import Path

import numpy as np
import voxelfuse as vf
from tqdm import tqdm

min_size = 5
membran_code = 1
bone_code = 2


def find_membran_surface(m_data, b_data):
    merge = m_data * membran_code + b_data * bone_code
    x_len, y_len, z_len = merge.shape
    regions = []
    for i in tqdm(range((x_len // min_size) + (1 if (x_len % min_size != 0) else 0))):
        for j in range((y_len // min_size) + (1 if (y_len % min_size != 0) else 0)):
            for k in range((z_len // min_size) + (1 if (z_len % min_size != 0) else 0)):
                slice = merge[i * min_size:(i + 1) * min_size,
                        j * min_size:(j + 1) * min_size,
                        k * min_size:(k + 1) * min_size]
                if np.sum(slice[slice == membran_code]) > 0:
                    regions.append((i, j, k, slice))
    print(len(regions))
    regions_border = list(filter(lambda x: x[3].mean() != membran_code, regions))
    print(len(regions_border))
    regions_border_without_bone = list(filter(lambda x: len(x[3][x[3] == bone_code]) == 0, regions))
    print(len(regions_border_without_bone))
    result_points = []
    for region in tqdm(regions_border_without_bone):
        i, j, k, slice = region
        for i_ in range(min_size):
            for j_ in range(min_size):
                for k_ in range(min_size):
                    i_i = i * min_size + i_ - 1
                    j_j = j * min_size + j_ - 1
                    k_k = k * min_size + k_ - 1
                    if i_i < 0 or j_j < 0 or k_k < 0 or i_i > (x_len - 3) or j_j > (y_len - 3) or k_k > (z_len - 3):
                        continue
                    slice_ = merge[i_i:i_i + 3,
                             j_j:j_j + 3,
                             k_k:k_k + 3]
                    if slice_.mean() < membran_code and slice_[1, 1, 1] == membran_code:
                        result_points.append((i_i + 1, j_j + 1, k_k + 1))
    print(len(regions_border_without_bone) * pow(min_size, 3))
    print(len(result_points))
    result_matrix = np.zeros(merge.shape)
    for i, j, k in tqdm(result_points):
        result_matrix[i, j, k] = 1

    return result_matrix


def convert2membran_object(membran_path: Path, bone_path: Path, path_output: Path):
    m_data = np.load(str(membran_path))
    m_data[m_data < 0.0] = 0.0
    m_data[m_data > 0.0] = 1.0
    b_data = np.load(str(bone_path))
    b_data[b_data < 0.0] = 0.0
    b_data[b_data > 0.0] = 1.0
    m_surf_data = find_membran_surface(m_data, b_data)
    model = vf.VoxelModel(m_surf_data, 1)
    save_numpy_as_mesh(model, path_output, True)


def save_numpy_as_mesh(voxel_model: vf.VoxelModel, path_output: Path, show: bool = False):
    mesh = vf.Mesh.fromVoxelModel(voxel_model, (0.0, 1.0, 0.0, 1.0))
    mesh.export(str(path_output))
    if show:
        mesh.viewer(grids=True, name='mesh')


def load_and_process(filepath_in: Path, filepath_out: Path):
    data = np.load(str(filepath_in))
    data[data < 0.0] = 0.0
    data[data > 0.0] = 1.0
    model = vf.VoxelModel(data, 1)
    save_numpy_as_mesh(model, filepath_out, True)


# Start Application
if __name__ == '__main__':
    file_in_m = Path("/media/kirrog/workdata/membransdata/newes/test/stpv/numpy/stpvM.npy")
    file_in_b = Path("/media/kirrog/workdata/membransdata/newes/test/stpv/numpy/stpvB.npy")
    file_out = Path("modelResult.stl")
    convert2membran_object(file_in_m, file_in_b, file_out)
