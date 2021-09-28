import sys

from stl import mesh

from src.stl_png.classes import vertex, triangle
from src.stl_png.transformer import stl2pngs
from src.utils.matrix2png_saver import transform_results

stl_bone_path = '/media/kirrog/data/dataset/stl/001//Bone.stl'


your_mesh = mesh.Mesh.from_file(stl_bone_path)
triangles = []
size = len(your_mesh.v0)
for i in range(size):
    v1 = vertex(your_mesh.v0[i][0], your_mesh.v0[i][1], your_mesh.v0[i][2])
    v2 = vertex(your_mesh.v1[i][0], your_mesh.v1[i][1], your_mesh.v1[i][2])
    v3 = vertex(your_mesh.v2[i][0], your_mesh.v2[i][1], your_mesh.v2[i][2])
    norm = vertex(your_mesh.normals[i][0], your_mesh.normals[i][1], your_mesh.normals[i][2])
    tr = triangle(v1, v2, v3, norm)
    triangles.append(tr)
    if i % 10000 == 0:
        sys.stdout.write("\rTriangle %i loaded" % i)
print('\nCreated')
images = stl2pngs(triangles, 512)
transform_results(images, 'stl/')
