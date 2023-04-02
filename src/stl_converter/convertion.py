from multiprocessing.pool import Pool

import numpy as np
from tqdm import tqdm


def create_sphere_array(size: int):
    data = np.zeros((size, size, size))
    for i in tqdm(range(data.shape[0])):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                x = i - (size // 2)
                y = j - (size // 2)
                z = k - (size // 2)
                if (x ** 2 + y ** 2 + z ** 2) < (size // 4) ** 2:
                    data[i, j, k] = 1
    print(data.shape)
    print(data.mean())
    return data


def convert_numpy2stl(data: np.array):
    assert len(data.shape) == 3
    x_size, y_size, z_size = data.shape
    surface_points = []
    for i in tqdm(range(data.shape[0])):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                val = data[max(i - 1, 0):min(i + 2, x_size - 1),
                      max(j - 1, 0):min(j + 2, y_size - 1),
                      max(k - 1, 0):min(k + 2, z_size - 1)].mean()
                if 1.0 > val > 0.0 and data[i, j, k] == 1:
                    surface_points.append((i, j, k))
    return surface_points


def extract_triangles(view):
    # кубиковые варианты - не будут получаться на внешней стороне - игнор
    # пирамидковые варианты
    # отсутствие одной точки
    # склоны пирамидки
    # плоскость
    # плоская пирамидка
    return []


def create_triangles(point, data):
    x_size, y_size, z_size = data.shape
    i, j, k = point
    view = data[max(i - 1, 0):min(i + 2, x_size - 1),
           max(j - 1, 0):min(j + 2, y_size - 1),
           max(k - 1, 0):min(k + 2, z_size - 1)]
    r = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                d = view[i:i + 2, j:j + 2, k:k + 2]
                r.append(d)
    result_triangles = []
    for v in r:
        result_triangles.extend(extract_triangles(v))
    return result_triangles


def protoype(data, points):
    with Pool() as p:
        triples = p.imap_unordered(create_triangles, [(x, data) for x in points])
    result = []
    for tr in triples:
        result.extend(tr)
    return result


size = 8
data = create_sphere_array(size)
points = convert_numpy2stl(data)
print(len(points))
print(points)
