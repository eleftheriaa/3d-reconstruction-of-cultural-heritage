from pathlib import Path
import yaml
import numpy as np
import open3d as o3d
import torch
 
# ============================================================
# LOAD PIPELINE
# ============================================================
 
# config_file = Path("nerf/instant_ngp_outputs/1_all_cone/instant-ngp/config.yml")
# config = yaml.load(config_file.read_text(), Loader=yaml.Loader)
# config.load_dir = Path("nerf/instant_ngp_outputs/1_all_cone/instant-ngp/nerfstudio_models/")
# config.print_to_terminal()
 
# trainer = config.setup(local_rank=0, world_size=1)
# trainer.setup()
 
# pipeline = trainer.pipeline
 
# ============================================================
# LOAD CLEANED POINT CLOUD
# ============================================================
 
pcd_path = Path(
    "nerf/pcds/instant-ngp/1_all_maskclean.ply"
)
 
print(f"Loading point cloud from: {pcd_path}")
pcd = o3d.io.read_point_cloud(str(pcd_path))
print(f"Loaded: {len(pcd.points):,} points")
 
# ============================================================
# AUTO SCALE PARAMETERS  (same logic as your pcd script)
# ============================================================
 
bbox   = pcd.get_axis_aligned_bounding_box()
extent = np.max(bbox.get_extent())
print(f"Scene extent: {extent}")
 
NORMAL_RADIUS = extent / 500
print(f"NORMAL_RADIUS: {NORMAL_RADIUS}")
 
# ============================================================
# RE-ESTIMATE NORMALS if not present
# (your save script sets estimate_normals=False so re-do here)
# ============================================================
 
if not pcd.has_normals():
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=30
        )
    )
    pcd.normalize_normals()
    pcd.orient_normals_consistent_tangent_plane(30)
    print("Normals ready.")
else:
    print("Normals already present.")
 
# ============================================================
# METHOD 1 — POISSON SURFACE RECONSTRUCTION
# ============================================================
 
print("\n--- Poisson reconstruction (depth=9) ---")
 
poisson_dir = Path("nerf/meshes/instant-ngp")
poisson_dir.mkdir(parents=True, exist_ok=True)
 
poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)
densities = np.asarray(densities)
 
# Remove low-density boundary floaters
keep = densities > np.quantile(densities, 0.1)
poisson_mesh.remove_vertices_by_mask(~keep)
 
print(f"Poisson mesh: {len(poisson_mesh.vertices):,} vertices, {len(poisson_mesh.triangles):,} faces")
 
poisson_out = poisson_dir / "1_all_mask_clean_poisson.ply"
o3d.io.write_triangle_mesh(str(poisson_out), poisson_mesh)
print(f"Saved → {poisson_out}")
 
torch.cuda.empty_cache()
 