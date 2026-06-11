import torch
from pathlib import Path

from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig

from nerfstudio.models.nerfacto import NerfactoModelConfig

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.trainer import TrainerConfig

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.configs.base_config import ViewerConfig

from nerfstudio.configs.method_configs import method_configs



from pathlib import Path

# Root dataset folder
root = "dataset"
subset = "All_faces_sculpted_derivatives"
example = "90_1920x1080_relief_heightmap_1_all_cone.obj"

# Combine into full path
datapath = Path(root) / subset / example

print("Full datapath:", datapath)

# Construct experiment name as subset/example
experiment_name = f"{subset}/{example}"

dataparser_config = ColmapDataParserConfig(
    data=datapath,
    scale_factor=1.0,
    scene_scale=1.5, 
    downscale_factor=1,
    load_3D_points=False,
    assume_colmap_world_coordinate_convention=False,
    orientation_method="none",
    center_method="none",
    auto_scale_poses=False,
    # eval_mode='all', 
    eval_mode="interval",
    eval_interval=8,
    )

config = TrainerConfig(
    experiment_name=experiment_name,
    project_name="shrec_1_nerfacto",
    method_name="nerfacto",
    timestamp = "",
    output_dir=Path("nerf/outputs/"),
    # steps_per_eval_batch=20,
    steps_per_eval_image=200,
    max_num_iterations=10000, 
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=dataparser_config,
            train_num_rays_per_batch=8192, 
            eval_num_rays_per_batch=8192,
        ),
        model=NerfactoModelConfig(
            near_plane=2,
            far_plane=6,
            proposal_initial_sampler="uniform",
            eval_num_rays_per_chunk=8192,
            average_init_density=0.07,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
            background_color="white", 
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=4096),
    vis="wandb",
)

config.save_config()

trainer = config.setup(local_rank=0, world_size=1)
trainer.setup()
trainer.train()


