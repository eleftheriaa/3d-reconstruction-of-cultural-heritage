from pathlib import Path
from create_saturation_masks import process_directory

dataset_root = Path(r"G:/elefth/NeRF/3d-reconstruction-of-cultural-heritage/dataset")

subsets = [
    "All_faces_sculpted_derivatives",
    "All_faces_sculpted_primitive",
]

for subset in subsets:
    subset_path = dataset_root / subset
    obj_folders = [f for f in subset_path.iterdir() if f.is_dir()]
    print(f"\n=== {subset} — {len(obj_folders)} objects ===")

    for obj_path in obj_folders:
        images_dir = obj_path / "images"
        masks_dir  = obj_path / "masks"

        if not images_dir.exists():
            print(f"  SKIP {obj_path.name} — no images/ folder")
            continue

        print(f"  Processing: {obj_path.name}")
        process_directory(
            input_dir=str(images_dir),
            output_dir=str(masks_dir),
            bg_threshold=0,
            sat_threshold=5,
            min_obj_width=10,
            erode=2,
            ext="png",
        )

print("\nAll done.")