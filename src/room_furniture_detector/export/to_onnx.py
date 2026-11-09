import shutil
import tempfile
from pathlib import Path

from room_furniture_detector import config


def export_onnx() -> Path:
    import onnx
    import onnxruntime as ort
    from ultralytics import YOLO

    source = config.MODEL_PT_PATH
    destination = config.MODEL_ONNX_PATH

    if not source.is_file():
        raise FileNotFoundError(f"Model weights not found: {source}")

    if config.EXPORT_IMGSZ <= 0:
        raise ValueError("EXPORT_IMGSZ must be positive.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        raise FileExistsError(
            f"Export destination already exists: {destination}. "
            "Choose another MODEL_ONNX_PATH or move the existing file."
        )

    with tempfile.TemporaryDirectory(prefix="furniture-export-") as directory:
        temporary_weights = Path(directory) / "model.pt"
        shutil.copy2(source, temporary_weights)

        model = YOLO(str(temporary_weights))

        exported = Path(
            model.export(
                format="onnx",
                imgsz=config.EXPORT_IMGSZ,
                batch=1,
                dynamic=False,
                simplify=False,
                opset=config.EXPORT_OPSET,
                device=config.EXPORT_DEVICE,
                nms=False,
            )
        )

        if not exported.is_file():
            raise RuntimeError(f"ONNX export not found: {exported}")

        onnx.checker.check_model(str(exported))

        session = ort.InferenceSession(
            str(exported),
            providers=["CPUExecutionProvider"],
        )

        for tensor in session.get_inputs():
            print(
                f"Input: name={tensor.name}, "
                f"shape={tensor.shape}, type={tensor.type}"
            )

        for tensor in session.get_outputs():
            print(
                f"Output: name={tensor.name}, "
                f"shape={tensor.shape}, type={tensor.type}"
            )

        shutil.copy2(exported, destination)

    print(f"ONNX model saved to: {destination}")
    return destination


def main() -> None:
    export_onnx()


if __name__ == "__main__":
    main()
