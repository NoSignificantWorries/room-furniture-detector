import csv
import random
from collections import Counter
from pathlib import Path

from omegaconf import DictConfig

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)


def get_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    images = sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise ValueError(f"No images found in {directory}")

    return images


def count_classes(
    images: list[Path],
    image_root: Path,
    label_root: Path,
    num_classes: int,
) -> Counter:
    counts = Counter()

    for image_path in images:
        relative_path = image_path.relative_to(image_root)
        label_path = (label_root / relative_path).with_suffix(".txt")

        if not label_path.is_file():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue

            values = line.split()
            if len(values) != 5:
                raise ValueError(
                    f"Invalid annotation: {label_path}:{line_number}"
                )

            class_id = int(values[0])
            if not 0 <= class_id < num_classes:
                raise ValueError(
                    f"Invalid class ID: {label_path}:{line_number}"
                )

            counts[class_id] += 1

    return counts


def prepare_splits(
    config: DictConfig,
    project_root: Path,
) -> dict[str, Path]:
    data_root = (project_root / config.data.root).resolve()
    source_train = (data_root / config.data.train).resolve()
    source_test = (data_root / config.data.val).resolve()

    images = get_images(source_train)
    test_images = get_images(source_test)

    val_fraction = float(config.data.split.val_fraction)
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    random.Random(config.data.split.seed).shuffle(images)

    val_size = round(len(images) * val_fraction)
    if not 0 < val_size < len(images):
        raise ValueError("Both train and validation must be non-empty.")

    split_images = {
        "train": sorted(images[val_size:]),
        "val": sorted(images[:val_size]),
        "test": test_images,
    }

    if set(split_images["train"]) & set(split_images["val"]):
        raise ValueError("Train and validation contain overlapping paths.")

    class_counts = {}
    for split_name, selected_images in split_images.items():
        image_root = source_test if split_name == "test" else source_train
        label_root = (
            data_root / "labels" / ("val" if split_name == "test" else "train")
        )

        counts = count_classes(
            selected_images,
            image_root,
            label_root,
            config.data.num_classes,
        )
        class_counts[split_name] = counts

        missing_classes = [
            name
            for class_id, name in enumerate(config.data.names)
            if counts[class_id] == 0
        ]
        if missing_classes:
            raise ValueError(
                f"Classes missing from {split_name}: {missing_classes}. "
                "Review the splitting strategy before training."
            )

    output_dir = (project_root / config.data.split.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for split_name, selected_images in split_images.items():
        manifest_path = output_dir / f"{split_name}.txt"
        manifest_path.write_text(
            "\n".join(path.as_posix() for path in selected_images) + "\n",
            encoding="utf-8",
        )
        manifests[split_name] = manifest_path
        print(f"{split_name}: {len(selected_images)} images")

    with (output_dir / "class_counts.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["split", *config.data.names])

        for split_name, counts in class_counts.items():
            writer.writerow(
                [
                    split_name,
                    *[
                        counts[class_id]
                        for class_id in range(config.data.num_classes)
                    ],
                ]
            )

    return manifests
