from pathlib import Path

import numpy as np

from src.modules.logger import CstmLogger

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


class Converter:
    membran_code = 1
    bone_code = 2

    def __init__(self, model_path: Path, cstm_logger: CstmLogger):
        self.cstm_logger = cstm_logger
        self.cstm_logger.log(f"Loading converter from: {model_path}")
        # self.model = keras.models.load_model(model_path)
        self.cstm_logger.log("Converter model loaded")

    def threshold_voxels_by_cohesion_number(self, data: np.array,
                                            component_elements_threshold: int = 100,
                                            voxels_neighbours_threshold: int = 3) -> np.array:
        x_len, y_len, z_len = data.shape
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
                        if i < 0 or j < 0 or k < 0 or i >= x_len or j >= y_len or k >= z_len:
                            continue
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
        return result_data

    def apply(self, segmented_datacube, defect_datacube, batch_size):
        self.cstm_logger.log(f"Start processing segment with shape: {segmented_datacube.shape}")
        self.cstm_logger.log(f"Start processing defect with shape: {defect_datacube.shape}")
        self.cstm_logger.log(f"Start processing with batch size: {batch_size}")
        merge = defect_datacube * self.membran_code + segmented_datacube * self.bone_code
        coords = np.where(merge == self.membran_code)
        regions_border_without_bone = []
        for i, j, k in zip(coords[0], coords[1], coords[2]):
            slice_data = merge[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]
            bone_slice_data = merge[i - 2:i + 3, j - 2:j + 3, k - 2:k + 3]
            if slice_data.mean() != self.membran_code and len(bone_slice_data[bone_slice_data == self.bone_code]) == 0:
                regions_border_without_bone.append((i, j, k))
        result_matrix = np.zeros(merge.shape)
        for i, j, k in regions_border_without_bone:
            result_matrix[i, j, k] = 1
        thresholded_matrix = self.threshold_voxels_by_cohesion_number(result_matrix)
        self.cstm_logger.log(f"Complete converting membran")
        return thresholded_matrix
