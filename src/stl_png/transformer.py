import math
import sys

import numpy as np

import src.stl_png.classes as cl

barrier = 0.5
x_size = 512
y_size = 512


def find_triangle_place(triangle, min_v, max_v, slices):
    step = float(max_v - min_v) / slices
    num1 = (triangle.v1.z - min_v) / step
    num2 = (triangle.v2.z - min_v) / step
    num3 = (triangle.v3.z - min_v) / step
    max_local = max(num1, num2, num3)
    min_local = min(num1, num2, num3)
    if min_local - int(min_local) > barrier:
        min_local = int(min_local) + 1
    else:
        min_local = int(min_local)
    if max_local - int(max_local) > barrier:
        max_local = int(max_local) + 1
    else:
        max_local = int(max_local)
    if max_local - min_local < 1:
        return -1
    else:
        return range(min_local, max_local)


def scale_norm(norm):
    k = norm.y / norm.x
    y = math.sqrt((pow(k, 2)) / (1 + pow(k, 2)))
    x = y / k
    return cl.vertex2d(x, y)


def get_lines(iterator, cycle):
    if len(cycle) - 1 == iterator:
        return cycle[iterator], cycle[0]
    else:
        return cycle[iterator], cycle[iterator + 1]


def is_lines_unitable(l1, l2, accuracy):
    normy = l1.norm.y + l2.norm.y
    normx = l1.norm.x + l2.norm.x
    norm = scale_norm(cl.vertex2d(normx, normy))
    vertx = norm.x * (l1.v2.x - l1.v1.x) + norm.y * (l1.v2.y - l1.v1.y)
    if vertx <= pow(0.1, accuracy - 1):
        return True, norm
    else:
        return False, norm


def shrink_lines_in_slice(cycle, accuracy):  # cycle throw the line
    lines_new = []
    iterator = 0
    without_change = 0
    size = len(cycle)
    while True:
        line_p, line_n = get_lines(iterator, cycle)
        bo, norm = is_lines_unitable(line_p, line_n, accuracy)
        if bo:
            l = cl.line(line_p.v1, line_n.v2, norm)
            line_p.v1.lines.remove(line_p)
            line_p.v2.lines.remove(line_p)
            line_n.v1.lines.remove(line_n)
            line_n.v2.lines.remove(line_n)
            lines_new.append(l)
            iterator += 2
        else:
            lines_new.append(line_p)
            without_change += 1
            iterator += 1
        if without_change == size:
            break
        if iterator >= size:
            without_change = 0
            iterator = 0
            cycle = lines_new
            lines_new = []
    return lines_new


def get_neighbour_lines_of_x2(line):
    t = line.v2.lines.copy()
    t.remove(line)
    return t


def find_figures(lines):  # from set of lines get time figures
    figures = []
    wrong_neigbours = 0
    dead_ends = 0
    check = np.zeros(len(lines))
    while True:
        iterator = -1
        for i, it in zip(check, range(len(lines))):
            if i == 0:
                iterator = it
                check[it] = 1
                break
        if iterator == -1:
            break
        fig = []
        start = lines[iterator]
        temp = start
        while True:
            t = get_neighbour_lines_of_x2(temp)
            if len(t) > 1:
                wrong_neigbours += len(t) - 1
            if len(t) == 0:
                dead_ends += 1
                break
            temp = t[0]
            ind = lines.index(temp)
            check[ind] = 1
            fig.append(temp)
            if temp == start:
                break
        figures.append(fig)
    if wrong_neigbours > 0:
        print("\nNumber of wrong neighbours " + str(wrong_neigbours))
    if dead_ends > 0:
        print("\nNumber of dead ends " + str(dead_ends))
    return figures


def make_figures_from_slice(z_lvl, triangles_slice, accuracy):
    lines = []
    size = len(triangles_slice)
    vertexes = []
    for i in range(size):
        line = get_triangle_slice_line(z_lvl, triangles_slice[i])
        vertexes.append(line.v1)
        vertexes.append(line.v2)
        lines.append(line)
        if i % 10000 == 0:
            sys.stdout.write("\rLines %i get" % i)
    vertexes.sort(key=lambda vert: vert.__hash__())
    vert = None
    res_vert = 0
    for i in vertexes:
        if type(vert) == type(None):
            vert = i
        else:
            if vert.__eq__(i):
                vert.lines.append(i.lines[0])
                i.lines[0].add_vertex2d(vert)
            else:
                res_vert += 1
                vert = i
    print("\nCreated " + str(len(lines)) + " with " + str(res_vert) + " vertexes")
    vertexes.clear()
    finded_lines = find_figures(lines)
    figures = []
    for fig in finded_lines:
        figures.append(cl.figure(shrink_lines_in_slice(fig, accuracy)))
    return figures


def get_point(z_lvl, v1, v2):
    universal = 0.0
    if v1.z - v2.z == 0.0:
        if z_lvl - v2.z != 0.0:
            return -1
        else:
            universal = 0.0
    else:
        universal = (z_lvl - v2.z) / (v1.z - v2.z)
    x = (universal * (v1.x - v2.x)) + v2.x
    y = (universal * (v1.y - v2.y)) + v2.y
    return cl.vertex2d(x, y)


def get_triangle_slice_line(z_lvl, triangle):  # add a check of v1 and v2 order
    points = get_triangle_slice_points(z_lvl, triangle)
    if len(points) > 1:
        v1 = cl.vertex2d(points[0].x, points[0].y)
        v2 = cl.vertex2d(points[1].x, points[1].y)
        norm = scale_norm(cl.vertex2d(triangle.normal.x, triangle.normal.y))
        return cl.line(v1, v2, norm)
    else:
        return 0


def get_triangle_slice_points(z_lvl, triangle):
    res = []
    if triangle.v1.z == z_lvl:
        res.append(triangle.v1)
    if triangle.v2.z == z_lvl:
        res.append(triangle.v2)
    if triangle.v3.z == z_lvl:
        res.append(triangle.v3)
    if len(res) >= 2:
        return res
    x_max = max(triangle.v1.x, triangle.v2.x, triangle.v3.x)
    x_min = min(triangle.v1.x, triangle.v2.x, triangle.v3.x)
    y_max = max(triangle.v1.y, triangle.v2.y, triangle.v3.y)
    y_min = min(triangle.v1.y, triangle.v2.y, triangle.v3.y)
    p1 = get_point(z_lvl, triangle.v1, triangle.v2)
    p2 = get_point(z_lvl, triangle.v2, triangle.v3)
    p3 = get_point(z_lvl, triangle.v1, triangle.v3)
    if isinstance(p1, cl.vertex2d):
        if x_min <= p1.x <= x_max and y_min <= p1.y <= y_max:
            res.append(p1)
    if isinstance(p2, cl.vertex2d):
        if x_min <= p2.x <= x_max and y_min <= p2.y <= y_max:
            res.append(p2)
    if isinstance(p3, cl.vertex2d):
        if x_min <= p3.x <= x_max and y_min <= p3.y <= y_max:
            res.append(p3)
    return list(set(res))


def figure_check(figure, x, y, x_scaler, y_scaler):
    minimal_dist = sys.maxsize * 2 + 1
    dist_res = -1
    for line in figure.lines:
        dist = (x - (line.x1 / x_scaler)) * line.normx + (y - (line.y1 / y_scaler)) * line.normy
        if minimal_dist > abs(dist):
            left_dist = pow(x - (line.v1.x / x_scaler), 2) + pow(y - (line.v1.y / y_scaler), 2)
            right_dist = pow(x - (line.v2.x / x_scaler), 2) + pow(y - (line.v2.y / y_scaler), 2)
            minimal_dist = abs(dist)
            dist_res = - dist
            if abs(dist) < left_dist and abs(dist) < right_dist:
                dist_res = -1
    return dist_res


def make_array_from_figure(array, figure, corr_x, corr_y, x_scaler, y_scaler):
    x_start = int(figure.x_min / x_scaler)
    x_stop = int(figure.x_max / x_scaler)
    y_start = int(figure.y_min / y_scaler)
    y_stop = int(figure.y_max / y_scaler)
    for i in range(x_start, x_stop):
        for j in range(y_start, y_stop):
            col = figure_check(figure, i - corr_x, j - corr_y, x_scaler, y_scaler)
            if col >= 0:
                array[i][j] = col
    return array


def get_scaler_variables(triangles, slices):
    max_x = sys.maxsize * (-2)
    max_y = sys.maxsize * (-2)
    max_z = sys.maxsize * (-2)
    min_x = sys.maxsize * 2 + 1
    min_y = sys.maxsize * 2 + 1
    min_z = sys.maxsize * 2 + 1
    for triangle in triangles:
        max_x = max(triangle.v1.x, triangle.v2.x, triangle.v3.x, max_x)
        max_y = max(triangle.v1.y, triangle.v2.y, triangle.v3.y, max_y)
        max_z = max(triangle.v1.z, triangle.v2.z, triangle.v3.z, max_z)
        min_x = min(triangle.v1.x, triangle.v2.x, triangle.v3.x, min_x)
        min_y = min(triangle.v1.y, triangle.v2.y, triangle.v3.y, min_y)
        min_z = min(triangle.v1.z, triangle.v2.z, triangle.v3.z, min_z)
    step = float(max_z - min_z) / slices
    x_scaler = (max_x - min_x) / x_size
    y_scaler = (max_y - min_y) / y_size
    accuracy = 0
    temperal_accuracy = math.sqrt(pow(x_scaler, 2) / 2 + pow(y_scaler, 2) / 2)
    t = 0
    while t == 0:
        accuracy += 1
        t = round(temperal_accuracy, accuracy)
    return min_z, max_z, x_scaler, y_scaler, accuracy, step


def stl2pngs(triangles, slices=512):
    min_z, max_z, x_scaler, y_scaler, accuracy, step = get_scaler_variables(triangles, slices)
    triangle_slices_numbers = []
    for i in range(slices):
        tri_slice_numbers = []
        triangle_slices_numbers.append(tri_slice_numbers)
    for triangle, i in zip(triangles, range(len(triangles))):
        res = find_triangle_place(triangle, min_z, max_z, slices)
        if res != -1:
            for r in res:
                triangle_slices_numbers[r].append(triangle)
        else:
            triangle.clear()
        if i % 10000 == 0:
            sys.stdout.write("\rTriangle %i sorted" % i)
    print()
    triangles.clear()
    result = []
    for i in range(slices):
        array = np.zeros((x_size, y_size), dtype=np.float16)
        z_lvl = step * (i + 0.5) + min_z
        figures = make_figures_from_slice(z_lvl, triangle_slices_numbers[i], accuracy)
        for fig in figures:
            array = make_array_from_figure(array, fig, x_size / 2, y_size / 2, x_scaler, y_scaler)
        result.append(array)
        sys.stdout.write("\rSlice %i created" % i)
    print()
    triangle_slices_numbers.clear()
    return result
