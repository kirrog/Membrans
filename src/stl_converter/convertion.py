from pathlib import Path

import numpy as np
import voxelfuse as vf

min_size = 3
membran_code = 1
bone_code = 2


def coordinate_hex(x: int, y: int, z: int) -> str:
    return hex(x + y * pow(10, 3) + z * pow(10, 6))


class Voxel:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.neighbours = set()
        self.marked = False

    def __hex__(self):
        return coordinate_hex(self.x, self.y, self.z)


def threshold_voxels_by_cohesion_number(data: np.array,
                                        component_elements_threshold: int = 100,
                                        voxels_neighbours_threshold: int = 3) -> np.array:
    hex2voxel = dict()
    coords = np.where(data == membran_code)
    for i, j, k in zip(coords[0], coords[1], coords[2]):
        hex2voxel[coordinate_hex(i, j, k)] = Voxel(i, j, k)
    for voxel in hex2voxel.values():
        for i_i in range(-1, 2):
            for j_j in range(-1, 2):
                for k_k in range(-1, 2):
                    if i_i == 0 and j_j == 0 and k_k == 0:
                        continue
                    i, j, k = voxel.x + i_i, voxel.y + j_j, voxel.z + k_k
                    if data[i, j, k] == membran_code:
                        neighbour = hex2voxel[coordinate_hex(i, j, k)]
                        voxel.neighbours.add(neighbour)
                        neighbour.neighbours.add(voxel)
    voxel2delete = list(filter(lambda x: len(x.neighbours) <= voxels_neighbours_threshold, hex2voxel.values()))
    for voxel in voxel2delete:
        for neighbour in voxel.neighbours:
            neighbour.neighbours.remove(voxel)
    voxels_set = {x for x in filter(lambda x: x not in voxel2delete, hex2voxel.values())}
    components = []
    while len(voxels_set) > 0:
        voxel = voxels_set.pop()
        component_elements = set()
        front = [voxel]
        while len(front) > 0:
            current_voxel = front.pop()
            if current_voxel in voxels_set:
                voxels_set.remove(current_voxel)
            current_voxel.marked = True
            component_elements.add(current_voxel)
            for elem in current_voxel.neighbours:
                if not elem.marked:
                    front.append(elem)
        components.append(list(component_elements))
    filtered_components = list(filter(lambda x: len(x) > component_elements_threshold, components))
    result_data = np.zeros(data.shape)
    counter = 0
    for component in filtered_components:
        for voxel in component:
            result_data[voxel.x, voxel.y, voxel.z] = 1
            counter += 1
    print(counter)
    return result_data


def find_membran_surface(m_data, b_data):
    merge = m_data * membran_code + b_data * bone_code
    coords = np.where(merge == membran_code)
    regions_border_without_bone = []
    for i, j, k in zip(coords[0], coords[1], coords[2]):
        slice_data = merge[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]
        if slice_data.mean() != membran_code and len(slice_data[slice_data == bone_code]) == 0:
            regions_border_without_bone.append((i, j, k))
    print(len(regions_border_without_bone))
    result_matrix = np.zeros(merge.shape)
    for i, j, k in regions_border_without_bone:
        result_matrix[i, j, k] = 1
    thresholded_matrix = threshold_voxels_by_cohesion_number(result_matrix)
    return thresholded_matrix


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
