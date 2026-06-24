from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from dataclasses import dataclass, field
from typing import Type, Dict, Tuple
import torch

class SplatfactoModelWithDepth(SplatfactoModel):
    def get_image_metrics_and_images(self, outputs, batch):
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        depth = outputs.get("depth", None)
        accumulation = outputs.get("accumulation", None)
        if depth is not None:
            images_dict["depth"] = self._apply_colormap(depth, accumulation)

        return metrics_dict, images_dict
    @staticmethod

    def _apply_colormap(depth: torch.Tensor, accumulation: torch.Tensor = None) -> torch.Tensor:
        import matplotlib.cm as cm

        depth_np = depth.squeeze(-1).detach().cpu().numpy()  # [H, W]

        if accumulation is not None:
            acc_np = accumulation.squeeze(-1).detach().cpu().numpy()
            valid_mask = acc_np > 0.01  # pixels that actually hit geometry
        else:
            valid_mask = np.ones_like(depth_np, dtype=bool)

        # normalize only over valid (foreground) depth values
        if valid_mask.any():
            valid_depths = depth_np[valid_mask]
            d_min, d_max = valid_depths.min(), valid_depths.max()
        else:
            d_min, d_max = 0.0, 1.0

        depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-8)
        depth_norm = depth_norm.clip(0, 1)

        colormap = cm.get_cmap("viridis")
        colored = colormap(depth_norm)[..., :3]

        # force background to white
        colored[~valid_mask] = [1.0, 1.0, 1.0]

        return torch.from_numpy(colored).to(depth.device).float()
@dataclass
class SplatfactoModelWithDepthConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoModelWithDepth)