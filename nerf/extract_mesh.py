import open3d as o3d
import numpy as np

# ============================================================
# LOAD CLEAN POINT CLOUD
# ============================================================

pcd = o3d.io.read_point_cloud(
    "pcds\All_faces_sculpted_primitive\90_1920x1080_relief_heightmap_1_all_clean.ply"
)

print(pcd)

if not pcd.has_normals():
    raise RuntimeError("Point cloud needs normals.")

# ============================================================
# ESTIMATE AVERAGE POINT SPACING
# ============================================================

distances = pcd.compute_nearest_neighbor_distance()

avg_dist = np.mean(distances)

print("Average point spacing:", avg_dist)

# ============================================================
# BALL PIVOTING RADII
# ============================================================

radii = [
    avg_dist * 1.5,
    avg_dist * 2.0,
    avg_dist * 3.0
]

print("Radii:", radii)

# ============================================================
# BALL PIVOTING RECONSTRUCTION
# ============================================================

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd,
    o3d.utility.DoubleVector(radii)
)

mesh.compute_vertex_normals()

# ============================================================
# SAVE
# ============================================================

o3d.io.write_triangle_mesh(
    "meshes\All_faces_sculpted_primitive\90_1920x1080_relief_heightmap_1_all_clean.ply",
    mesh
)

# ============================================================
# VISUALIZE
# ============================================================

o3d.visualization.draw_geometries(
    [mesh],
    mesh_show_back_face=True
)