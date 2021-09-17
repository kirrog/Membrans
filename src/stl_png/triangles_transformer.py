import sys

from stl import mesh

from src.stl_png.classes import vertex, triangle


def load_triangles_from_stl(path):
    your_mesh = mesh.Mesh.from_file(path)
    triangles = []
    size = len(your_mesh.v0)
    vertexes = []
    for i in range(size):
        v1 = vertex(your_mesh.v0[i][0], your_mesh.v0[i][1], your_mesh.v0[i][2])
        v2 = vertex(your_mesh.v1[i][0], your_mesh.v1[i][1], your_mesh.v1[i][2])
        v3 = vertex(your_mesh.v2[i][0], your_mesh.v2[i][1], your_mesh.v2[i][2])
        vertexes.append(v1)
        vertexes.append(v2)
        vertexes.append(v3)
        norm = vertex(your_mesh.normals[i][0], your_mesh.normals[i][1], your_mesh.normals[i][2])
        tr = triangle(v1, v2, v3, norm)
        triangles.append(tr)
        if i % 10000 == 0:
            sys.stdout.write("\rTriangle %i loaded" % i)
    vertexes.sort(key=lambda vert: vert.__hash__())
    vert = None
    res_vert = 0
    for i in vertexes:
        if type(vert) == type(None):
            vert = i
        else:
            if vert.__eq__(i):
                vert.triangles.append(i.triangles[0])
                i.triangles[0].add_vertex(vert)
            else:
                res_vert += 1
                vert = i
    print("\nCreated + " + str(len(triangles)) + " with " + str(res_vert) + " vertexes" )
    vertexes.clear()
    return triangles
