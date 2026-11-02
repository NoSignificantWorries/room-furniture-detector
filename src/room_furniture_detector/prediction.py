import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def predict_image(
    image: str,
    weights: str,
    output: str,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
    save_image: bool = False,
) -> None:
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1.")

    if imgsz <= 0:
        raise ValueError("imgsz must be positive.")

    project_root = Path(__file__).resolve().parents[2]
    image_path = _resolve_path(image, project_root)
    weights_path = _resolve_path(weights, project_root)
    output_path = _resolve_path(output, project_root)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    if output_path.suffix.lower() != ".json":
        raise ValueError("Output file must have a .json extension.")

    annotated_path = output_path.with_name(f"{output_path.stem}_pred.jpg")

    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    if save_image and annotated_path.exists():
        raise FileExistsError(
            f"Annotated image already exists: {annotated_path}"
        )

    if save_image and annotated_path == image_path:
        raise ValueError("Annotated image must not overwrite the input image.")

    try:
        with Image.open(image_path) as source:
            input_image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot read image: {image_path}") from exc

    model = YOLO(str(weights_path))
    if model.task != "detect":
        raise ValueError("The checkpoint must be an object detection model.")

    result = model.predict(
        source=input_image,
        conf=conf,
        imgsz=imgsz,
        device=str(device),
        save=False,
        verbose=False,
    )[0]

    height, width = result.orig_shape
    detections = []

    if result.boxes is not None:
        boxes = result.boxes.cpu()
        for coordinates, confidence, class_id in zip(
            boxes.xyxy.tolist(),
            boxes.conf.tolist(),
            boxes.cls.tolist(),
            strict=True,
        ):
            class_id = int(class_id)
            x_min, y_min, x_max, y_max = coordinates
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": float(confidence),
                    "bbox": {
                        "x_min": float(x_min),
                        "y_min": float(y_min),
                        "x_max": float(x_max),
                        "y_max": float(y_max),
                    },
                }
            )

    payload = {
        "schema_version": "1.0",
        "image": {
            "filename": image_path.name,
            "width": int(width),
            "height": int(height),
        },
        "model": {
            "checkpoint": weights_path.name,
            "confidence_threshold": conf,
            "imgsz": imgsz,
        },
        "bbox_format": "xyxy",
        "coordinate_system": "original_image_pixels",
        "detections": detections,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_image:
        result.save(filename=str(annotated_path))
        print(f"Annotated image: {annotated_path}")

    with output_path.open("x", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")

    print(f"Detected objects: {len(detections)}")
    print(f"JSON saved to: {output_path}")
