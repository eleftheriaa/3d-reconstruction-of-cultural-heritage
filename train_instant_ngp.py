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

# Root dataset folder
root = Path("dataset")
subset = "All_faces_sculpted_primitive"

# Folder containing all objects
subset_path = root / subset

# Find all object folders
object_folders = [f for f in subset_path.iterdir() if f.is_dir()]

print(f"Found {len(object_folders)} objects")

for obj_path in object_folders:

    print(f"\nTraining object: {obj_path.name}")

    datapath = obj_path

    experiment_name = f"{obj_path.name}"

    # --- Dataparser ---
    dataparser_config = ColmapDataParserConfig(
        data=datapath,
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
        experiment_name=experiment_name,
        project_name="shrec",
        method_name=f"{subset}",
        steps_per_eval_image=1000,
        steps_per_save=1000,
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
                    max_steps=20000
                ),
            }
        },
    )

    # --- Setup and train ---
    config.set_timestamp()
    config.pipeline.model.device = "cuda"
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