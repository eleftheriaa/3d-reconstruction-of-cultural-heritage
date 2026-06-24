from pathlib import Path

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from splatfacto_depth import SplatfactoModelWithDepthConfig
# --- Dataparser ---
# NOTE: splatfacto works best with COLMAP points for Gaussian initialization.
# The ColmapDataParserConfig will automatically load 3D points if available.

root = Path("dataset")
subset = "All_faces_sculpted_derivatives"
subset_short_name = "1_d"

# Folder containing all objects
subset_path = root / subset

# Find all object folders
object_folders = [f for f in subset_path.iterdir() if f.is_dir()]

print(f"Found {len(object_folders)} objects")

for obj_path in object_folders:

    print(f"\nTraining object: {obj_path.name}")

    datapath = obj_path
    short_name = "_".join(obj_path.name.replace(".obj", "").split("_")[4:])

    experiment_name = f"{subset_short_name}" + f"/{short_name}"


    dataparser_config = ColmapDataParserConfig(
        data=datapath,
        images_path = Path("images_lama"),
        scale_factor=1.0,
        scene_scale=1.0,
        downscale_factor=1,
        load_3D_points=True,           # important: seeds Gaussians from SfM points
        orientation_method="none",
        center_method="none",
        auto_scale_poses=False,
    )
    # --- Full config ---
    config = TrainerConfig(
        experiment_name= experiment_name  ,
        method_name= "splatfacto",
        project_name = "splatfacto_1",
        timestamp = "lama",
        # steps_per_eval_batch=0,
        steps_per_eval_image=500,
        steps_per_save=1000,
        # steps_per_eval_all_images=1000,
        max_num_iterations=15000,
        mixed_precision=False,
        use_grad_scaler=False,
        save_only_latest_checkpoint=True,
        vis="wandb",


        pipeline=VanillaPipelineConfig(

            datamanager=FullImageDatamanagerConfig(
                dataparser=dataparser_config,
                masks_on_gpu=False,
                images_on_gpu=False,
                camera_res_scale_factor=1.0,
                eval_num_images_to_sample_from=-1,
                eval_num_times_to_repeat_images=-1,
                eval_image_indices=(0,),
                cache_images="gpu",
                cache_images_type="uint8",
                max_thread_workers=None,
                train_cameras_sampling_strategy="random",
                train_cameras_sampling_seed=42,
                fps_reset_every=100,
            ),

            model=SplatfactoModelWithDepthConfig(
                warmup_length=500,
                refine_every=100,
                resolution_schedule=3000,
                background_color="white",
                num_downscales=2,
                cull_alpha_thresh=0.005,
                cull_scale_thresh=0.5,
                reset_alpha_every=30,
                densify_grad_thresh=0.0006,
                densify_size_thresh=0.01,
                n_split_samples=2,
                sh_degree_interval=1000,
                cull_screen_size=0.15,
                split_screen_size=0.05,
                stop_screen_size_at=4000,
                random_init=False,
                # num_random=50000,
                # random_scale=10.0,
                ssim_lambda=0.2,
                stop_split_at=30000,
                sh_degree=3,
                use_scale_regularization=True,
                max_gauss_ratio=10.0,
                output_depth_during_training=True,
                rasterize_mode="classic",          # or "antialiased"
                camera_optimizer=CameraOptimizerConfig(
                    mode="off",                    # set to "SO3xR3" to enable camera opt
                    trans_l2_penalty=0.01,
                    rot_l2_penalty=0.001,
                ),
                use_bilateral_grid=False,
                grid_shape=(16, 16, 8),
                color_corrected_metrics=False,
            ),
        ),

        # --- Per-parameter optimizers (Gaussian Splatting has separate LRs per attribute) ---
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=1e-8,
                    lr_final=1.6e-6,
                    warmup_steps=0,
                    max_steps=30000,
                    ramp="cosine",
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=2.5e-3, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=1.25e-4, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=5e-2, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0,
                    lr_final=5e-7,
                    warmup_steps=1000,
                    max_steps=30000,
                    ramp="cosine",
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0,
                    lr_final=1e-4,
                    warmup_steps=1000,
                    max_steps=30000,
                    ramp="cosine",
                ),
            },
        },
    )

    # --- Setup and train ---
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




# using the cli

# ns-train splatfacto --project-name splatfacto_1 --method_name splatfacto --experiment-name 1_d/1_all_sphere --timestamp "" --pipeline.model.background-color white --pipeline.model.random-init True colmap --data dataset/All_faces_sculpted_derivatives/90_1920x1080_relief_heightmap_1_all_sphere.obj --load-3D-points False --auto-scale-poses False --downscale-factor 1

# ns-render dataset  --load-config gaussian\outputs\All_faces_sculpted_derivatives\1_all_cone_mask\splatfacto\config.yml --output-path gaussian/renders_m --rendered-output-names depth 

# ns-export gaussian-splat --load-config splatfacto/All_faces_sculpted_primitive/90_1920x1080_relief_heightmap_1_all.obj/config.yml --output-dir exports/splatfacto/All_faces_sculpted_primitive/90_1920x1080_relief_heightmap_1_all.obj