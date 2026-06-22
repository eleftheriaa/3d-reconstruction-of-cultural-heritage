from pathlib import Path
import shutil
import subprocess
import sys

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("dataset/All_faces_sculpted_derivatives")

COLMAP_EXE = "colmap"  # Or full path, e.g. r"C:\Program Files\COLMAP\COLMAP.bat"

CAMERA_MODEL = "SIMPLE_PINHOLE"
MAX_FEATURES = 20000

# True = also create sparse.ply beside images/ and colmap/
EXPORT_PLY = True

# True = process every object folder ending in .obj
PROCESS_ALL = True

# Set this only if PROCESS_ALL = False
SINGLE_OBJECT = Path(
    "dataset/All_faces_sculpted_derivatives/"
    "90_1920x1080_relief_heightmap_1_all_cone.obj"
)

# ============================================================


def run_command(command: list[str]) -> None:
    print("\nRunning:")
    print(" ".join(f'"{x}"' if " " in x else x for x in command))

    result = subprocess.run(command, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"\nCOLMAP command failed with exit code {result.returncode}:\n"
            + " ".join(command)
        )


def ensure_valid_empty_points3d(points_path: Path) -> None:
    """
    COLMAP needs the input sparse model to contain cameras.txt,
    images.txt, and points3D.txt. This creates a valid empty
    points3D.txt only if it is missing or zero bytes.
    """
    if points_path.exists() and points_path.stat().st_size > 0:
        return

    points_path.write_text(
        "# 3D point list with one line of data per point:\n"
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as "
        "(IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0\n",
        encoding="utf-8",
    )

    print(f"Created valid empty model file: {points_path}")


def process_object(obj_dir: Path) -> None:
    images_dir = obj_dir / "images"
    sparse0_dir = obj_dir / "colmap" / "sparse" / "0"

    cameras_txt = sparse0_dir / "cameras.txt"
    images_txt = sparse0_dir / "images.txt"
    points3d_txt = sparse0_dir / "points3D.txt"

    database_path = obj_dir / "colmap" / "database.db"
    triangulated_dir = obj_dir / "colmap" / "triangulated_temp"

    # PLY goes beside images/ and colmap/
    output_ply = obj_dir / "sparse.ply"

    print("\n" + "=" * 70)
    print(f"Processing: {obj_dir.name}")
    print("=" * 70)

    if not images_dir.is_dir():
        print(f"SKIP: images folder missing: {images_dir}")
        return

    if not cameras_txt.exists():
        print(f"SKIP: cameras.txt missing: {cameras_txt}")
        return

    if not images_txt.exists():
        print(f"SKIP: images.txt missing: {images_txt}")
        return

    # Make empty points3D.txt loadable by COLMAP.
    ensure_valid_empty_points3d(points3d_txt)

    # Remove old database so feature extraction starts clean.
    if database_path.exists():
        database_path.unlink()
        print(f"Removed old database: {database_path}")

    # Remove temporary triangulation output.
    if triangulated_dir.exists():
        shutil.rmtree(triangulated_dir)

    triangulated_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Extract SIFT features
    # --------------------------------------------------------
    run_command([
        COLMAP_EXE,
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", CAMERA_MODEL,
        "--SiftExtraction.max_num_features", str(MAX_FEATURES),
    ])

    # --------------------------------------------------------
    # 2. Match all images
    # --------------------------------------------------------
    run_command([
        COLMAP_EXE,
        "exhaustive_matcher",
        "--database_path", str(database_path),
    ])

    # --------------------------------------------------------
    # 3. Triangulate using existing cameras.txt + images.txt
    # --------------------------------------------------------
    run_command([
        COLMAP_EXE,
        "point_triangulator",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--input_path", str(sparse0_dir),
        "--output_path", str(triangulated_dir),
    ])

    # COLMAP may write text or binary depending on build/config.
    new_points_txt = triangulated_dir / "points3D.txt"
    new_points_bin = triangulated_dir / "points3D.bin"

    if new_points_txt.exists():
        # Backup original empty / previous points file.
        backup = sparse0_dir / "points3D_before_triangulation.txt"
        if points3d_txt.exists() and not backup.exists():
            shutil.copy2(points3d_txt, backup)

        shutil.copy2(new_points_txt, points3d_txt)
        print(f"\nReplaced seed cloud: {points3d_txt}")

    elif new_points_bin.exists():
        # If COLMAP produced binary output, copy all 3 bin files.
        for filename in ["cameras.bin", "images.bin", "points3D.bin"]:
            source = triangulated_dir / filename
            destination = sparse0_dir / filename

            if source.exists():
                shutil.copy2(source, destination)

        print(f"\nCopied binary sparse model into: {sparse0_dir}")

    else:
        raise RuntimeError(
            "Triangulation completed but no points3D.txt or points3D.bin was produced."
        )

    # --------------------------------------------------------
    # 4. Export sparse.ply beside images/ and colmap/
    # --------------------------------------------------------
    if EXPORT_PLY:
        if output_ply.exists():
            output_ply.unlink()

        run_command([
            COLMAP_EXE,
            "model_converter",
            "--input_path", str(triangulated_dir),
            "--output_path", str(output_ply),
            "--output_type", "PLY",
        ])

        print(f"Created PLY: {output_ply}")

    # Optional cleanup: keep it False initially for debugging.
    # shutil.rmtree(triangulated_dir)

    print(f"\nFinished: {obj_dir.name}")


def main():
    if PROCESS_ALL:
        object_dirs = sorted(
            p for p in DATASET_ROOT.iterdir()
            if p.is_dir() and p.name.endswith(".obj")
        )
    else:
        object_dirs = [SINGLE_OBJECT]

    if not object_dirs:
        print(f"No object folders found in: {DATASET_ROOT}")
        sys.exit(1)

    print(f"Found {len(object_dirs)} object folders.")

    failures = []

    for obj_dir in object_dirs:
        try:
            process_object(obj_dir)
        except Exception as exc:
            print(f"\nFAILED: {obj_dir.name}")
            print(exc)
            failures.append(obj_dir.name)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    if failures:
        print(f"Failed objects ({len(failures)}):")
        for name in failures:
            print(f"  - {name}")
    else:
        print("All objects completed successfully.")


if __name__ == "__main__":
    main()