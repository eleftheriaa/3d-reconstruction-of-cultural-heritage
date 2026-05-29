import argparse
from pathlib import Path
import yaml
import numpy as np

# 2026-04-02_001758
# outputs\instant-ngp-scene-scale05-grid-1\2048\2026-05-21_164829
# G:\elefth\shrec\eythimis\outputs\dataset\nerfacto\2026-04-02_110514\config.yml
config_file = Path("outputs/All_faces_sculpted_primitive/90_1920x1080_relief_heightmap_1_all.obj/instant_ngp_tuned/2026-05-28_173743/config.yml")
config = yaml.load(config_file.read_text(), Loader=yaml.Loader)   
config.load_dir = Path("outputs/All_faces_sculpted_primitive/90_1920x1080_relief_heightmap_1_all.obj/instant_ngp_tuned/2026-05-28_173743/nerfstudio_models")
config.load_step = None
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
    pipeline=trainer.pipeline,
    num_points=1000000,
    remove_outliers=True,
    estimate_normals=False,
    reorient_normals=False,
    rgb_output_name="rgb",
    depth_output_name="depth",
    crop_obb=crop_obb,
    std_ratio=10,
)

output_path = Path("pcd_box/All_faces_sculpted_primitive/90_1920x1080_relief_heightmap_1_all.obj.ply")
o3d.io.write_point_cloud(str(output_path), pcd)

print(f"Point cloud saved to {output_path}")