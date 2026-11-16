import hashlib
import json
import re
import tempfile
from pathlib import Path

from room_furniture_detector import config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _download_file(
    destination: Path,
    file_id: str,
    expected_sha256: str,
) -> Path:
    import gdown

    expected_sha256 = expected_sha256.strip().lower()

    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"Invalid expected SHA-256 for: {destination}")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
        raise ValueError(f"Invalid Google Drive file ID for: {destination}")

    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"Expected a file: {destination}")

        actual = _sha256(destination)

        if actual != expected_sha256:
            raise ValueError(
                f"Existing file checksum mismatch: {destination}\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual}\n"
                "Move the existing file or choose another destination."
            )

        print(f"Already verified: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".model-download-",
        dir=destination.parent,
    ) as directory:
        temporary = Path(directory) / "artifact.part"

        result = gdown.download(
            id=file_id,
            output=str(temporary),
            quiet=False,
            use_cookies=False,
        )

        if result is None or not temporary.is_file():
            raise RuntimeError(
                f"Google Drive download failed: {destination.name}"
            )

        actual = _sha256(temporary)

        if actual != expected_sha256:
            raise ValueError(
                f"Downloaded file checksum mismatch: {destination.name}\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual}"
            )

        if destination.exists():
            raise FileExistsError(
                f"Destination appeared during download: {destination}"
            )

        temporary.replace(destination)

    print(f"Downloaded and verified: {destination}")
    return destination


def download_models(backend: str = "onnx") -> list[Path]:
    artifacts = {
        "pt": (
            config.MODEL_PT_PATH,
            config.MODEL_PT_GDRIVE_ID,
            config.MODEL_PT_SHA256,
        ),
        "onnx": (
            config.MODEL_ONNX_PATH,
            config.MODEL_ONNX_GDRIVE_ID,
            config.MODEL_ONNX_SHA256,
        ),
        "tensorrt": (
            config.MODEL_TENSORRT_PATH,
            config.MODEL_TENSORRT_GDRIVE_ID,
            config.MODEL_TENSORRT_SHA256,
        ),
        "tensorrt_metadata": (
            config.MODEL_TENSORRT_METADATA_PATH,
            config.MODEL_TENSORRT_METADATA_GDRIVE_ID,
            config.MODEL_TENSORRT_METADATA_SHA256,
        ),
    }

    selections = {
        "pt": ("pt",),
        "onnx": ("onnx",),
        "tensorrt": ("tensorrt", "tensorrt_metadata"),
        "all": ("pt", "onnx", "tensorrt", "tensorrt_metadata"),
    }

    if backend not in selections:
        raise ValueError("backend must be one of: pt, onnx, tensorrt, all.")

    selected = selections[backend]
    paths = [artifacts[name][0].resolve() for name in selected]

    if len(paths) != len(set(paths)):
        raise ValueError("Artifact destination paths must be distinct.")

    downloaded = [_download_file(*artifacts[name]) for name in selected]

    if "tensorrt" in selected:
        metadata = json.loads(
            config.MODEL_TENSORRT_METADATA_PATH.read_text(encoding="utf-8")
        )

        if (
            metadata.get("engine_sha256")
            != config.MODEL_TENSORRT_SHA256.strip().lower()
        ):
            raise ValueError(
                "TensorRT metadata does not match the configured engine."
            )

        if (
            metadata.get("source_onnx_sha256")
            != config.MODEL_ONNX_SHA256.strip().lower()
        ):
            raise ValueError(
                "TensorRT metadata references a different ONNX model."
            )

    return downloaded


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and verify model artifacts from Google Drive."
    )
    parser.add_argument(
        "--backend",
        choices=("pt", "onnx", "tensorrt", "all"),
        default="onnx",
    )

    args = parser.parse_args()
    download_models(backend=args.backend)


if __name__ == "__main__":
    main()
