"""
create_saturation_masks.py
--------------------------
Generates binary masks for saturated (255,255,255) regions on a geometric object
photographed against a white background.

Mask output:
  - WHITE (255) : saturated pixels INSIDE the object (excluding perimeter)
  - BLACK (0)   : background + non-saturated object pixels

Usage:
    python create_saturation_masks.py --input ./images --output ./masks
    python improvement/masks/create_saturated_masks.py --input dataset\All_faces_sculpted_derivatives\90_1920x1080_relief_heightmap_1_all_cone.obj\images --output dataset\All_faces_sculpted_derivatives\90_1920x1080_relief_heightmap_1_all_cone.obj\saturated_masks
Optional flags:
    --bg_threshold   How close to white counts as "background" (default: 0)
    --sat_threshold  How close to 255 on all channels counts as "saturated" (default: 5)
    --min_obj_width  Min pixel span to count a row as containing the object (default: 10)
    --erode          Pixels to erode inward from object boundary (default: 2)
    --ext            Input image extension (default: png)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_object_mask_vectorized(img_array: np.ndarray,
                                 bg_threshold: int = 10,
                                 min_obj_width: int = 10) -> np.ndarray:
    """
    Fully vectorized row-wise left/right scan.
    Returns boolean array (H, W) — True where pixel is INSIDE the object silhouette.
    """
    # True where pixel is background-white
    is_white = np.all(img_array >= (255 - bg_threshold), axis=2)   # (H, W)
    is_obj   = ~is_white                                             # (H, W)

    H, W = is_obj.shape
    cols  = np.arange(W, dtype=np.int32)

    # For each row: left boundary = first True col in is_obj
    # argmax on bool finds first True; if all False → returns 0 (handle separately)
    any_obj = is_obj.any(axis=1)                                     # (H,)

    left_idx  = np.argmax(is_obj, axis=1)                           # (H,)
    right_idx = W - 1 - np.argmax(is_obj[:, ::-1], axis=1)         # (H,)

    span = right_idx - left_idx                                      # (H,)

    # Build fill mask: for each row fill [left, right] inclusive
    obj_mask = (
        (cols[None, :] >= left_idx[:, None]) &
        (cols[None, :] <= right_idx[:, None]) &
        any_obj[:, None] &
        (span[:, None] >= min_obj_width)
    )

    return obj_mask


def create_mask(img_array: np.ndarray,
                bg_threshold: int = 10,
                sat_threshold: int = 20,
                min_obj_width: int = 10,
                erode: int = 2) -> np.ndarray:
    """
    Returns uint8 mask (H, W):
      255 → saturated pixel strictly inside the object (boundary stripped)
        0 → everything else
    """
    obj_mask = find_object_mask_vectorized(img_array, bg_threshold, min_obj_width)

    # Erode to remove perimeter — prevents bg-adjacent white pixels bleeding in
    if erode > 0:
        struct = np.ones((erode * 2 + 1, erode * 2 + 1), dtype=bool)
        obj_mask = binary_erosion(obj_mask, structure=struct)

    sat_mask = np.all(img_array >= (255 - sat_threshold), axis=2)

    return (obj_mask & sat_mask).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def process_directory(input_dir: str,
                       output_dir: str,
                       bg_threshold: int = 10,
                       sat_threshold: int = 20,
                       min_obj_width: int = 10,
                       erode: int = 2,
                       ext: str = "png"):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = sorted(input_path.glob(f"*.{ext.lower()}"))
    if not images:
        images = sorted(input_path.glob(f"*.{ext.upper()}"))
    if not images:
        print(f"[ERROR] No .{ext} files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images  |  erode={erode}px  bg_thr={bg_threshold}  sat_thr={sat_threshold}")

    for img_path in tqdm(images, unit="img"):
        img       = Image.open(img_path).convert("RGB")
        img_array = np.array(img, dtype=np.uint8)
        mask      = create_mask(img_array, bg_threshold, sat_threshold, min_obj_width, erode)

        out_file = output_path / (img_path.stem + ".png")
        Image.fromarray(mask, mode="L").save(out_file)

    print(f"Done! Masks saved to: {output_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate saturation masks for specular highlights on geometric objects."
    )
    p.add_argument("--input",         required=True, help="Input image directory")
    p.add_argument("--output",        required=True, help="Output mask directory")
    p.add_argument("--bg_threshold",  type=int, default=10,
                   help="Tolerance for background white detection (default: 10)")
    p.add_argument("--sat_threshold", type=int, default=20,
                   help="Tolerance for saturated pixel detection (default: 20)")
    p.add_argument("--min_obj_width", type=int, default=10,
                   help="Min pixel span per row to be considered object (default: 10)")
    p.add_argument("--erode",         type=int, default=2,
                   help="Pixels to erode inward from object boundary (default: 2)")
    p.add_argument("--ext",           default="png",
                   help="Image extension to look for (default: png)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(
        input_dir=args.input,
        output_dir=args.output,
        bg_threshold=args.bg_threshold,
        sat_threshold=args.sat_threshold,
        min_obj_width=args.min_obj_width,
        erode=args.erode,
        ext=args.ext,
    )
