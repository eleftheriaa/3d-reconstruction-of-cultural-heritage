import numpy as np
import torch
import yaml
from pathlib import Path
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.scripts.exporter import ExportGaussianSplat


config_file = Path("gaussian/outputs/All_faces_sculpted_derivatives_90_1920x1080_relief_heightmap_1_all_cone.obj/splatfacto/config.yml")
print("Looking for:", config_file)
print("Absolute path:", config_file.resolve())
print("Exists:", config_file.exists())
config = yaml.load(config_file.read_text(), Loader=yaml.Loader)   
# config.load_dir = Path("gaussian/outputs/All_faces_sculpted_derivatives_90_1920x1080_relief_heightmap_1_all_cone.obj/splatfacto/nerfstudio_models")
# config.load_step = None
config.print_to_terminal()    

trainer = config.setup(local_rank=0, world_size=1)
trainer.setup()

pipeline = trainer.pipeline
# --- Compute object center from training cameras ---
cameras = trainer.pipeline.datamanager.train_dataset.cameras
positions    = cameras.camera_to_worlds[:, :3, 3].numpy()
forward_dirs = -cameras.camera_to_worlds[:, :3, 2].numpy()
dots         = np.sum(-positions * forward_dirs, axis=1, keepdims=True)
closest_pts  = positions + dots * forward_dirs
object_center = closest_pts.mean(axis=0)
print(f"Object center: {object_center}")

# --- Convert to obb_* tuples that ExportGaussianSplat expects ---
half_extent = np.array([0.53,0.53,0.53]) * 2.0  # half extent = (1.0, 1.0, 1.0) for a 2m box

obb_center   = tuple(float(v) for v in object_center)       # (cx, cy, cz)
obb_rotation = (0.0, 0.0, 0.0)                              # no rotation (RPY = 0)
obb_scale    = tuple(float(v) for v in half_extent )     # full extent = (2.0, 2.0, 2.0)

# --- Run the exporter ---
exporter = ExportGaussianSplat(
    output_filename= "1_all_cone_sh_coeffs.ply",
    load_config=trainer.config.get_base_dir() / "config.yml",
    output_dir=Path("gaussian\exports\splats"),
    obb_center=obb_center,
    obb_rotation=obb_rotation,
    obb_scale=obb_scale,
    ply_color_mode = "sh_coeffs",
)
exporter.main()
print("Export complete! saved to:", exporter.output_dir)