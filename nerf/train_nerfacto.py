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
root = Path("dataset")
subset = "All_faces_sculpted_derivatives"
subset_short_name = "1_d"
example = ""

subset_path = root / subset
object_folders = [f for f in subset_path.iterdir() if f.is_dir()]




print(f"Found {len(object_folders)} objects")

for obj_path in object_folders:

    print(f"\nTraining object: {obj_path.name}")

    datapath = obj_path
    obj_short_name = "_".join(obj_path.name.replace(".obj", "").split("_")[4:])

    experiment_name = f"{subset_short_name}" + f"/{obj_short_name}"

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
        images_path=Path("images_lama"),
        # eval_mode='all', 
        eval_mode="interval",
        eval_interval=8,
        )

    config = TrainerConfig(
        experiment_name=experiment_name,
        project_name="nerfacto_1",
        method_name="nerfacto",
        timestamp = "",
        output_dir=Path("outputs/nerfacto/lama"),
        # steps_per_eval_batch=20,
        steps_per_eval_image=500,
        max_num_iterations=10000, 
        save_only_latest_checkpoint=False,
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
                average_init_density=0.01,
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

    # --- Setup and train ---
    # config.set_timestamp()
    config.pipeline.model.device = "cuda"
    config.save_config()

    trainer = config.setup(local_rank=0, world_size=1)
    trainer.setup()

    # import torch

    # cameras = trainer.pipeline.datamanager.train_dataset.cameras
    # positions = cameras.camera_to_worlds[:, :3, 3]
    # dists = torch.norm(positions, dim=-1)

    # print(
    #     f"Camera distances: "
    #     f"min={dists.min():.3f} "
    #     f"max={dists.max():.3f}"
    # )

    trainer.train()

# ns-render dataset --load-config nerf\nerfacto_outputs\All_faces_sculpted_derivatives\1_all_cone\nerfacto\config.yml --output-path nerf/nerfacto_renders --rendered-output-names rgb depth