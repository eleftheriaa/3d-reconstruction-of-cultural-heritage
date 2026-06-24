from simple_lama_inpainting import SimpleLama
from PIL import Image
from pathlib import Path

simple_lama = SimpleLama()  # downloads pretrained weights automatically

dataset_jpath = Path("dataset")
subset = "All_faces_sculpted_derivatives"
root = dataset_jpath / subset
object_folders = [f for f in root.iterdir() if f.is_dir()]
print("Found {} objects".format(len(object_folders)))



for obj_path in object_folders:
    print(f"\nProcessing object: {obj_path.name}")
    images_dir = obj_path / "images"
    masks_dir  = obj_path / "masks"
    output_dir = obj_path / "images_lama"
    output_dir.mkdir(exist_ok=True)
    for img_path in sorted(images_dir.glob("*.png")):
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")
        result = simple_lama(image, mask)
        result.save(output_dir / img_path.name)
        print(f"Done: {img_path.name}")