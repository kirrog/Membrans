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
    return cl.vertex(x, y, z_lvl)


def get_triangle_slice_line(z_lvl, triangle, accuracy):
    points = get_triangle_slice_points(z_lvl, triangle)
    if len(points) > 1:
        return cl.line(round(points[0].x, accuracy), round(points[0].y, accuracy), round(points[1].x, accuracy),
                       round(points[1].y, accuracy), round(triangle.normal.x, accuracy),
                       round(triangle.normal.y, accuracy))
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
    if p1 != -1:
        if x_min <= p1.x <= x_max and y_min <= p1.y <= y_max:
            res.append(p1)
    if p2 != -1:
        if x_min <= p2.x <= x_max and y_min <= p2.y <= y_max:
            res.append(p2)
    if p3 != -1:
        if x_min <= p3.x <= x_max and y_min <= p3.y <= y_max:
            res.append(p3)
    return list(set(res))


def figure_check(figure, x, y, x_scaler, y_scaler):
    minimal_dist = sys.maxsize * 2 + 1
    dist_res = -1
    for line in figure.lines:
        dist = (x - (line.x1 / x_scaler)) * line.normx + (y - (line.y1 / y_scaler)) * line.normy
        if minimal_dist > abs(dist):
            left_dist = pow(x - (line.x1 / x_scaler), 2) + pow(y - (line.y1 / y_scaler), 2)
            right_dist = pow(x - (line.x2 / x_scaler), 2) + pow(y - (line.y2 / y_scaler), 2)
            minimal_dist = abs(dist)
            dist_res = - dist
            if abs(dist) < left_dist and abs(dist) < right_dist:
                dist_res = -1
    return dist_res


def make_array_from_figure(array, figure, corr_x, corr_y, x_scaler, y_scaler):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            col = figure_check(figure, i - corr_x, j - corr_y, x_scaler, y_scaler)
            if col >= 0:
                array[i][j] = col
    return array


def lines_neighbors(l1, l2):
    if pow(l1.x1 - l2.x2, 2) + pow(l1.y1 - l2.y2, 2) == 0.0:
        return -1
    if pow(l1.x2 - l2.x1, 2) + pow(l1.y2 - l2.y1, 2) == 0.0:
        return -1
    return 0


def form_figures_from_lines_slow(lines):
    figures = []
    numbers = np.zeros(len(lines))
    work = True
    while work:
        result = []
        iterator = -1
        for i, it in zip(numbers, range(numbers.size)):
            if i != 1:
                iterator = it
                numbers[it] = 1
                break
        if iterator == -1:
            work = False
            break
        l = lines[iterator]
        result.append(l)
        added = [l]
        added_size = 1
        while added_size > 0:
            ad_next = []
            added_size_next = 0
            for line in added:
                for i, it in zip(numbers, range(len(numbers))):
                    if i != 1 and lines_neighbors(line, lines[it]) != 0:
                        result.append(lines[it])
                        ad_next.append(lines[it])
                        added_size_next += 1
                        numbers[it] = 1
            added = ad_next
            added_size = added_size_next
        figures.append(cl.figure(result))
        found = False
        for i in numbers:
            if i == 0:
                found = True
                break
        work = found
    return figures


def form_figures_from_lines(lines):
    x1_line = sorted(lines, key=lambda line: line.x1)
    x2_line = sorted(lines, key=lambda line: line.x2)
    figures = []
    numbers = np.zeros((len(lines), 2))
    work = True
    while work:
        result = []
        iterator_x1 = -1
        for it in range(len(lines)):
            if numbers[it][0] != 1:
                iterator_x1 = it
                break
        if iterator_x1 == -1:
            work = False
            break
        added_size = 1
        while added_size > 0:
            l = x1_line[iterator_x1]
            result.append(l)
            numbers[iterator_x1][0] = 1
            added_size_next = 0
            iterator_x2 = iterator_x1
            for line_ind in range(iterator_x2, -1, -1):  # searching start of equals x2
                if x2_line[line_ind].x2 < l.x1:
                    if iterator_x2 > line_ind >= 0:
                        iterator_x2 = line_ind + 1
                    break
                if line_ind == 0:
                    iterator_x2 = line_ind
            for line, key in zip(x2_line[iterator_x2:], range(iterator_x2, numbers.size)):
                if numbers[key][1] != 1 and line.x2 == l.x1 and line.y2 == l.y1:
                    numbers[key][1] = 1
                    iterator_x1 = x1_line.index(line)
                    if numbers[iterator_x1][0] == 1:
                        added_size_next = -1
                    else:
                        added_size_next = 1
                    break
            if added_size_next == 0:
                # it may need corrections
                print('Found uncycled varient')
                fir = result[0]
                last = result[len(result) - 1]
                middle = line(fir.x2, fir.y2, last.x1, last.y1, fir.normx + last.normx, fir.normy + last.normy)
                result.append(middle)
                numbers[x2_line.index(fir)][1] = 1
                numbers[x1_line.index(last)][0] = 1
            if added_size_next == -1:
                added_size_next = 0
            added_size = added_size_next
        figures.append(cl.figure(result))
        sys.stdout.write("\rFigure %i appended" % len(figures))
    return figures


def stl2pngs(triangles, slices=512):
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
    triangle_slices_numbers = []
    for i in range(slices):
        tri_slice_numbers = []
        triangle_slices_numbers.append(tri_slice_numbers)
    for triangle, i in zip(triangles, range(len(triangles))):
        res = (find_triangle_place(triangle, min_z, max_z, slices))
        if res != -1:
            for r in res:
                triangle_slices_numbers[r].append(triangle)
        if i % 10000 == 0:
            sys.stdout.write("\rTriangle %i sorted" % i)
    print()
    triangles.clear()
    slices_points = []
    for i in range(slices):
        slice = []
        z_lvl = step * (i + 0.5) + min_z
        for triangle in triangle_slices_numbers[i]:
            res = get_triangle_slice_line(z_lvl, triangle, accuracy)
            if res != 0:
                slice.append(res)
        triangle_slices_numbers[i].clear()
        slices_points.append(slice)
        sys.stdout.write("\rSlice %i collected" % i)
    print()
    triangle_slices_numbers.clear()
    result = []
    for i in range(slices):
        array = np.zeros((x_size, y_size), dtype=np.float16)
        figures = form_figures_from_lines(slices_points[i])
        slices_points[i].clear()
        for fig in figures:
            array = make_array_from_figure(array, fig, x_size / 2, y_size / 2, x_scaler, y_scaler)
        result.append(array)
        sys.stdout.write("\rSlice %i created" % i)
    print()
    slices_points.clear()
    return result
