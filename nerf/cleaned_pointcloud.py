import argparse
from pathlib import Path
import yaml
import numpy as np

config_file = Path("nerf/nerfacto_outputs/All_faces_sculpted_derivatives/1_all_cone/nerfacto/config.yml")
config = yaml.load(config_file.read_text(), Loader=yaml.Loader)   
config.load_dir = Path("nerf/nerfacto_outputs/All_faces_sculpted_derivatives/1_all_cone/nerfacto/nerfstudio_models/")
# config.load_step = None
config.print_to_terminal()    

trainer = config.setup(local_rank=0, world_size=1)
trainer.setup()

pipeline = trainer.pipeline


from nerfstudio.exporter.exporter_utils import generate_point_cloud

import open3d as o3d
from nerfstudio.data.scene_box import OrientedBox
import torch

# --- Get object center from cameras ---
cameras = trainer.pipeline.datamanager.train_dataset.cameras
positions = cameras.camera_to_worlds[:, :3, 3].numpy()
forward_dirs = -cameras.camera_to_worlds[:, :3, 2].numpy()
dots = np.sum(-positions * forward_dirs, axis=1, keepdims=True)
closest_points = positions + dots * forward_dirs
object_center = closest_points.mean(axis=0)
print(f"Object center: {object_center}")

# --- Build OrientedBox (axis-aligned, no rotation) ---
half_extent = np.array([0.53, 0.53, 0.53])  # same as your bbox logic

crop_obb = OrientedBox(
    R=torch.eye(3),                                        # no rotation = axis aligned
    T=torch.tensor(object_center, dtype=torch.float32),   # center
    S=torch.tensor(half_extent * 2, dtype=torch.float32), # full extent (not half)
)
# --- Generate point cloud with crop ---
pcd = generate_point_cloud(
    pipeline=pipeline,
    num_points=1000000,
    remove_outliers=True,
    estimate_normals=False,
    reorient_normals=False,
    rgb_output_name="rgb",
    depth_output_name="depth",
    crop_obb=crop_obb,
    std_ratio=10,
)

print("Initial point cloud:")
print(pcd)

# ============================================================
# CLEANING PIPELINE
# ============================================================
# ============================================================
# AUTO SCALE PARAMETERS
# ============================================================

bbox = pcd.get_axis_aligned_bounding_box()
extent = np.max(bbox.get_extent())

print("Scene extent:", extent)

VOXEL_SIZE = extent / 800
NORMAL_RADIUS = extent / 500
DBSCAN_EPS = extent / 100
RADIUS_OUTLIER = extent / 80

print("VOXEL_SIZE:", VOXEL_SIZE)
print("NORMAL_RADIUS:", NORMAL_RADIUS)
print("DBSCAN_EPS:", DBSCAN_EPS)

# ============================================================
# DOWNSAMPLE
# ============================================================

pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

print("After voxel:", len(pcd.points))

# ============================================================
# STATISTICAL OUTLIER REMOVAL
# ============================================================

pcd, ind = pcd.remove_statistical_outlier(
    nb_neighbors=10,
    std_ratio=10
)

print("After statistical:", len(pcd.points))

# ============================================================
# RADIUS OUTLIER REMOVAL
# ============================================================

pcd, ind = pcd.remove_radius_outlier(
    nb_points=10,
    radius=RADIUS_OUTLIER
)

print("After radius:", len(pcd.points))

# ============================================================
# CHECK EMPTY
# ============================================================

if len(pcd.points) == 0:
    raise RuntimeError(
        "Point cloud became empty after filtering."
    )

# ============================================================
# NORMALS
# ============================================================

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=NORMAL_RADIUS,
        max_nn=30
    )
)

pcd.normalize_normals()

print("Normals estimated.")

# ============================================================
# ORIENT NORMALS
# ============================================================

pcd.orient_normals_consistent_tangent_plane(30)

print("Normals oriented.")

# ============================================================
# DBSCAN
# ============================================================

labels = np.array(
    pcd.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=15,
        print_progress=True
    )
)

max_label = labels.max()

print("Clusters found:", max_label + 1)

if max_label >= 0:

    cluster_sizes = [
        np.sum(labels == i)
        for i in range(max_label + 1)
    ]

    largest_cluster = np.argmax(cluster_sizes)

    indices = np.where(labels == largest_cluster)[0]

    pcd = pcd.select_by_index(indices)

    print("After DBSCAN:", len(pcd.points))
else:
    print("No DBSCAN clusters found.")
#============================================================
# SAVE CLEANED POINT CLOUD
# ============================================================

output_path = Path(
    "nerf/pcds/nerfacto/"
    "1_all_cone_clean.ply"
)

o3d.io.write_point_cloud(str(output_path), pcd)

print(f"Cleaned point cloud saved to {output_path}")

