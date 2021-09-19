from src.stl_png.transformer import stl2pngs
from src.stl_png.triangles_transformer import load_triangles_from_stl
from src.utils.results_transformer import transform_results

stl_bone_path = '../dataset/stl/001//Bone.stl'

triangles = load_triangles_from_stl(stl_bone_path)
images = stl2pngs(triangles, 49)
transform_results(images, 'stl/')
