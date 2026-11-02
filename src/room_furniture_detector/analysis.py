from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image

matplotlib.use("Agg")

import matplotlib.pyplot as plt

CLASS_NAMES = [
    "bed",
    "sofa",
    "chair",
    "table",
    "lamp",
    "tv",
    "laptop",
    "wardrobe",
    "window",
    "door",
    "potted plant",
    "photo frame",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_image_stats(split_dir: Path) -> list[dict]:
    records = []

    for image_path in split_dir.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        records.append(
            {
                "split": split_dir.name,
                "image_name": image_path.name,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 3),
            }
        )

    return records


def collect_label_stats(
    split_dir: Path,
    labels_dir: Path,
) -> list[dict]:
    records = []

    for label_path in labels_dir.rglob("*.txt"):
        image_stem = label_path.stem
        image_path = next(
            (
                candidate
                for candidate in split_dir.rglob(f"{image_stem}.*")
                if candidate.suffix.lower() in IMAGE_EXTENSIONS
            ),
            None,
        )

        if image_path is None:
            continue

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        with label_path.open(encoding="utf-8") as label_file:
            for line_number, line in enumerate(label_file, start=1):
                values = line.strip().split()

                if len(values) != 5:
                    continue

                class_id = int(values[0])
                box_width = float(values[3])
                box_height = float(values[4])

                records.append(
                    {
                        "split": split_dir.name,
                        "image_name": image_path.name,
                        "label_file": label_path.name,
                        "line_number": line_number,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "bbox_width": box_width,
                        "bbox_height": box_height,
                        "bbox_area": box_width * box_height,
                    }
                )

    return records


def save_class_counts(annotations: pd.DataFrame, output_dir: Path) -> None:
    counts = (
        annotations.groupby(["split", "class_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASS_NAMES, fill_value=0)
    )

    counts.to_csv(output_dir / "class_counts.csv")

    axes = counts.plot(kind="bar", figsize=(14, 6))
    axes.set_title("Object count by class")
    axes.set_ylabel("Number of objects")
    axes.set_xlabel("Dataset split")
    plt.tight_layout()
    plt.savefig(output_dir / "class_counts.png", dpi=200)
    plt.close()


def save_bbox_area_stats(annotations: pd.DataFrame, output_dir: Path) -> None:
    stats = (
        annotations.groupby(["split", "class_name"])["bbox_area"]
        .describe()
        .reset_index()
    )

    stats.to_csv(output_dir / "bbox_area_stats.csv", index=False)

    plt.figure(figsize=(14, 6))
    data_by_class = [
        annotations.loc[annotations["class_name"] == class_name, "bbox_area"]
        for class_name in CLASS_NAMES
    ]

    plt.boxplot(data_by_class, tick_labels=CLASS_NAMES, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Normalized bbox area")
    plt.title("Bounding box area by class")
    plt.tight_layout()
    plt.savefig(output_dir / "bbox_area_by_class.png", dpi=200)
    plt.close()


def save_image_size_stats(images: pd.DataFrame, output_dir: Path) -> None:
    images.to_csv(output_dir / "image_sizes.csv", index=False)

    plt.figure(figsize=(10, 6))
    for split_name, group in images.groupby("split"):
        plt.scatter(
            group["width"], group["height"], s=8, alpha=0.5, label=split_name
        )

    plt.xlabel("Image width, px")
    plt.ylabel("Image height, px")
    plt.title("Image size distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "image_sizes.png", dpi=200)
    plt.close()


def analyze_dataset(
    data_dir: str = "data",
    output_dir: str = "reports/dataset",
) -> None:
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_records = []
    annotation_records = []

    for split_name in ("train", "val"):
        split_dir = data_path / "images" / split_name
        labels_dir = data_path / "labels" / split_name

        image_records.extend(collect_image_stats(split_dir))
        annotation_records.extend(collect_label_stats(split_dir, labels_dir))

    images = pd.DataFrame(image_records)
    annotations = pd.DataFrame(annotation_records)

    if images.empty or annotations.empty:
        raise RuntimeError(
            "Dataset is empty. Check data/images and data/labels directories."
        )

    save_image_size_stats(images, output_path)
    save_class_counts(annotations, output_path)
    save_bbox_area_stats(annotations, output_path)

    summary = {
        "train_images": int((images["split"] == "train").sum()),
        "val_images": int((images["split"] == "val").sum()),
        "train_annotations": int((annotations["split"] == "train").sum()),
        "val_annotations": int((annotations["split"] == "val").sum()),
    }

    pd.Series(summary).to_csv(output_path / "summary.csv", header=["value"])
    print(summary)
