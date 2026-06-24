# fix_viewer.py
import sys
import time
from pathlib import Path

def patch_queue_size():
    import nerfstudio.data.datamanagers.parallel_datamanager as pdm
    original_init = pdm.ParallelDataManager.__init__

    def patched_init(self, config, **kwargs):
        if not hasattr(config, 'queue_size'):
            config.queue_size = 4
        original_init(self, config, **kwargs)

    pdm.ParallelDataManager.__init__ = patched_init

def patch_and_launch(config_path: str):
    patch_queue_size()

    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.viewer.viewer import Viewer
    import nerfstudio.configs.base_config as base_cfg

    config, pipeline, _, step = eval_setup(
        Path(config_path),
        eval_num_rays_per_chunk=None,
        test_mode="inference",
    )

    # Override viewer config to make sure it binds
    config.viewer.websocket_port = 7007
    config.viewer.websocket_host = "0.0.0.0"

    viewer = Viewer(
        config.viewer,
        log_filename=Path("viewer_log.txt"),
        datapath=pipeline.datamanager.get_datapath(),
        pipeline=pipeline,
        trainer=None,
        train_lock=None,
    )

    print(f"\n✓ Viewer running at: http://viewer.nerf.studio/?websocket_url=ws://localhost:7007")
    print("  (or open viewer.nerf.studio and enter ws://localhost:7007 manually)\n")

    # Keep the viewer alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else \
        r"outputs/1_d/1_all_cone/instant-ngp/config.yml"
    patch_and_launch(config_path)