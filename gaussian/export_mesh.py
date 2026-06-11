from nerfstudio.scripts.exporter import ExportGaussianSplat, ExportPoissonMesh
import yaml

import numpy as np
import torch
import open3d as o3d
from pathlib import Path
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.exporter.exporter_utils import generate_point_cloud, get_mesh_from_filename
from nerfstudio.exporter import texture_utils
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE




config_file = Path("gaussian/outputs/All_faces_sculpted_derivatives_90_1920x1080_relief_heightmap_1_all_cone.obj/splatfacto/config.yml")
print("Looking for:", config_file)
print("Absolute path:", config_file.resolve())
print("Exists:", config_file.exists())
config = yaml.load(config_file.read_text(), Loader=yaml.Loader)   
config.load_dir = Path("gaussian/outputs/All_faces_sculpted_derivatives_90_1920x1080_relief_heightmap_1_all_cone.obj/splatfacto/nerfstudio_models")
# config.load_step = None
config.print_to_terminal()    

trainer = config.setup(local_rank=0, world_size=1)
trainer.setup()

pipeline = trainer.pipeline



# ---------------------------------------------------------------------------
# Compute object center from training cameras
# ---------------------------------------------------------------------------
cameras = trainer.pipeline.datamanager.train_dataset.cameras
positions    = cameras.camera_to_worlds[:, :3, 3].numpy()
forward_dirs = -cameras.camera_to_worlds[:, :3, 2].numpy()
dots         = np.sum(-positions * forward_dirs, axis=1, keepdims=True)
closest_pts  = positions + dots * forward_dirs
object_center = closest_pts.mean(axis=0)
print(f"Object center: {object_center}")

# ---------------------------------------------------------------------------
# Shared OBB config (used by both exporters)
# ---------------------------------------------------------------------------
half_extent = np.array([1.0, 1.0, 1.0])

obb_center   = tuple(float(v) for v in object_center)    # (cx, cy, cz)
obb_rotation = (0.0, 0.0, 0.0)                           # axis-aligned, no rotation
obb_scale    = tuple(float(v) for v in half_extent * 2)  # full extent = (2.0, 2.0, 2.0)

config_path  = trainer.config.get_base_dir() / "config.yml"
output_dir=Path("gaussian\exports\All_faces_sculpted_derivatives_90_1920x1080_relief_heightmap_1_all_cone.obj")
output_dir_mesh=Path("gaussian\exports\All_faces_sculpted_derivatives_90_1920x1080_relief_heightmap_1_all_cone.obj\mesh")

# ---------------------------------------------------------------------------
# 1. Export Gaussian Splat (.ply)
# ---------------------------------------------------------------------------
ExportGaussianSplat(
    load_config=config_path,
    output_dir=output_dir,
    output_filename="splat.ply",
    obb_center=obb_center,
    obb_rotation=obb_rotation,
    obb_scale=obb_scale,
).main()
print("Export complete! saved to:", output_dir)




# ---------------------------------------------------------------------------
# 2. Export Poisson Mesh — manually, bypassing the VanillaDataManager assert
# ---------------------------------------------------------------------------
CONSOLE.print("Generating point cloud for Poisson meshing...")
 
pcd = generate_point_cloud(
    pipeline=trainer.pipeline,
    num_points=1_000_000,
    remove_outliers=True,
    reorient_normals=True,
    estimate_normals=True,        # open3d normals — splatfacto has no normal output
    rgb_output_name="rgb",
    depth_output_name="depth",
    normal_output_name=None,      # no model normals for splatfacto
    crop_obb=crop_obb,
    std_ratio=10.0,
)
torch.cuda.empty_cache()
CONSOLE.print(f"Generated point cloud: {pcd}")
 
# Optionally save the point cloud
o3d.io.write_point_cloud(str(output_dir / "point_cloud.ply"), pcd)
CONSOLE.print("Saved point_cloud.ply")
 
# Poisson reconstruction
CONSOLE.print("Running Poisson surface reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
vertices_to_remove = densities < np.quantile(densities, 0.1)
mesh.remove_vertices_by_mask(vertices_to_remove)
 
o3d.io.write_triangle_mesh(str(output_dir / "poisson_mesh.ply"), mesh)
CONSOLE.print("Saved poisson_mesh.ply")
 
# Texture the mesh by re-querying the NeRF
CONSOLE.print("Texturing mesh with NeRF...")
mesh_for_texture = get_mesh_from_filename(
    str(output_dir / "poisson_mesh.ply"),
    target_num_faces=50_000,
)
texture_utils.export_textured_mesh(
    mesh_for_texture,
    trainer.pipeline,
    output_dir,
    px_per_uv_triangle=None,      # None = use xatlas
    unwrap_method="xatlas",
    num_pixels_per_side=2048,
)
CONSOLE.print("Done — textured mesh exported to exports/my_splat/")