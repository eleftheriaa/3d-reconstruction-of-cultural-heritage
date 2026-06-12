# load_dir=Path("outputs\instant-ngp\2048\2026-05-21_102256\nerfstudio_models"),
# load_step=5000,
# C:\Users\vvr\anaconda3\Scripts\activate.bat nerfstudio
# G:
# cd elefth/shrec/instant-ngp

from pathlib import Path

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig


# --- Dataparser ---
dataparser_config = ColmapDataParserConfig(
    data=Path("dataset/All_faces_sculpted_derivatives/90_1920x1080_relief_heightmap_1_all_cone.obj"),
    scale_factor=1.0,
    scene_scale=1.0,
    downscale_factor=1,
    load_3D_points=False,
    assume_colmap_world_coordinate_convention=False,
    orientation_method="none",
    center_method="none",
    auto_scale_poses=False,
    eval_mode="interval",
    eval_interval=8,
)

# --- Full config ---
config = TrainerConfig(
    experiment_name="All_faces_sculpted_derivatives/1_all_cone",
    project_name="shrec",
    method_name= "instant-ngp",
    timestamp = "",
    output_dir=Path("nerf/instant_ngp_outputs/"),
    steps_per_eval_image=500,
    steps_per_save=500,
    save_only_latest_checkpoint=False,
    max_num_iterations=15000,
    mixed_precision=True,
    vis="wandb",

    viewer=ViewerConfig(
        num_rays_per_chunk=2048,
    ),

    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=dataparser_config,
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
        ),

        model=InstantNGPModelConfig(
            eval_num_rays_per_chunk=2048,
            near_plane=3.5,
            far_plane=5.5,
            grid_resolution=128,
            disable_scene_contraction=False,
            grid_levels=1,
            max_res=1024,
            log2_hashmap_size=19,
            alpha_thre=0.001,
            cone_angle=0.0,
            use_gradient_scaling=False,
            background_color="white",
        ),
    ),

    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(
                lr=1e-2,
                eps=1e-15
            ),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-4,
                max_steps=15000
            ),
        }
    },
)

# --- Setup and train ---
# config.set_timestamp()
config.pipeline.model.device = "cuda"
config.pipeline.datamanager.dataparser.data = Path("dataset/All_faces_sculpted_derivatives/90_1920x1080_relief_heightmap_1_all_cone.obj")  # redundant but explicit
config.save_config()

trainer = config.setup(local_rank=0, world_size=1)
trainer.setup()

import torch

cameras = trainer.pipeline.datamanager.train_dataset.cameras
positions = cameras.camera_to_worlds[:, :3, 3]
dists = torch.norm(positions, dim=-1)

print(
    f"Camera distances: "
    f"min={dists.min():.3f} "
    f"max={dists.max():.3f}"
)

trainer.train()