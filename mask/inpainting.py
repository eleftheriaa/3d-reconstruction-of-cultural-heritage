import cv2
import numpy as np
from pathlib import Path
import argparse


def inpaint_image(image, mask):
    """
    image: BGR image (H,W,3)
    mask: grayscale mask (H,W), 0=keep, >0=inpaint
    """
    return cv2.inpaint(
        image,
        mask,
        3,
        cv2.INPAINT_TELEA
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Inpaint saturated regions in images."
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output directory for inpainted images"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(90):

        image_path = (
            "dataset/All_faces_sculpted_derivatives/"
            "90_1920x1080_relief_heightmap_1_all_cone.obj/"
            f"images/frame_{i+1:03d}.png"
        )

        mask_path = (
            "dataset/All_faces_sculpted_derivatives/"
            "90_1920x1080_relief_heightmap_1_all_cone.obj/"
            f"saturated_masks/frame_{i+1:03d}.png"
        )

        image = cv2.imread(image_path)

        # IMPORTANT: load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Frame {i+1:03d}: image not found")
            continue

        if mask is None:
            print(f"Frame {i+1:03d}: mask not found")
            continue

        num_masked = np.count_nonzero(mask)

        if num_masked == 0:
            print(f"Frame {i+1:03d}: empty mask")
            cv2.imwrite(
                str(output_path / f"frame_{i+1:03d}.png"),
                image
            )
            continue


        inpainted_image = inpaint_image(image, mask)

        # OPTIONAL:
        # guarantee no masked pixel remains at 255
        inpainted_image[mask > 0] = np.minimum(
            inpainted_image[mask > 0],
            235
        )

        max_masked_value = inpainted_image[mask > 0].max()

        print(
            f"Frame {i+1:03d}: "
            f"max pixel value in masked region = {max_masked_value} ,"
            f"masked={num_masked}, "

        )

        cv2.imwrite(
            str(output_path / f"frame_{i+1:03d}.png"),
            inpainted_image
        )