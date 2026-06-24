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
        "--input",
        required=True,
        help="Input directory containing images and masks"
    )
    p.add_argument(
        "--masks",
        required=True,
        default="saturated_masks",
        help="Suffix for mask files (default: _saturated_mask.png)"
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

    input_path = Path(args.input)
    masks_path = Path(args.masks)

    print(f"Input images: {input_path}")
    print(f"Input masks: {masks_path}")

    for i in range(90):
        single_image_path = input_path / f"frame_{i+1:03d}.png"
        single_mask_path = masks_path / f"frame_{i+1:03d}.png"

        # print(single_image_path)
        # print(single_mask_path)
        image = cv2.imread(single_image_path)

        # IMPORTANT: load mask as grayscale
        mask = cv2.imread(single_mask_path, cv2.IMREAD_GRAYSCALE)

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
            245
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