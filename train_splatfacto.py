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

# --- Dataparser ---
# NOTE: splatfacto works best with COLMAP points for Gaussian initialization.
# The ColmapDataParserConfig will automatically load 3D points if available.
dataparser_config = ColmapDataParserConfig(
    data=Path("cube"),
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
    method_name="splatfacto",
    experiment_name=None,
    steps_per_save=2000,
    steps_per_eval_batch=0,
    steps_per_eval_image=100,
    steps_per_eval_all_images=1000,
    max_num_iterations=30000,
    mixed_precision=False,
    use_grad_scaler=False,
    save_only_latest_checkpoint=True,
    vis="viewer",

    viewer=ViewerConfig(
        websocket_port_default=7007,
        num_rays_per_chunk=32768,
        max_num_display_images=512,
        quit_on_train_completion=False,
        image_format="jpeg",
        jpeg_quality=75,
        make_share_url=False,
        camera_frustum_scale=0.1,
        default_composite_depth=True,
    ),

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

        model=SplatfactoModelConfig(
            warmup_length=500,
            refine_every=100,
            resolution_schedule=3000,
            background_color="random",
            num_downscales=2,
            cull_alpha_thresh=0.005,
            cull_scale_thresh=0.5,
            continue_cull_post_densification=False,
            reset_alpha_every=30,
            densify_grad_thresh=0.0006,
            densify_size_thresh=0.01,
            n_split_samples=2,
            sh_degree_interval=1000,
            cull_screen_size=0.15,
            split_screen_size=0.05,
            stop_screen_size_at=4000,
            random_init=False,
            num_random=50000,
            random_scale=10.0,
            ssim_lambda=0.2,
            stop_split_at=30000,
            sh_degree=3,
            use_scale_regularization=True,
            max_gauss_ratio=10.0,
            output_depth_during_training=False,
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
config.set_timestamp()
config.pipeline.datamanager.dataparser.data = Path("cube")
config.save_config()

trainer = config.setup(local_rank=0, world_size=1)
trainer.setup()
trainer.train()
