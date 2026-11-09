import hashlib
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from room_furniture_detector import config


def _extract_dataset(archive_path: Path, data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    root = data_root.resolve()

    with ZipFile(archive_path) as archive:
        members = []

        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            relative = PurePosixPath(name)

            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive entry: {member.filename}")

            if not relative.parts:
                continue

            if relative.parts[0] not in {"images", "labels"}:
                continue

            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(
                    f"Archive symlinks are not supported: {member.filename}"
                )

            destination = root.joinpath(*relative.parts).resolve()

            if not destination.is_relative_to(root):
                raise ValueError(f"Unsafe archive entry: {member.filename}")

            members.append((member, destination))

        if not members:
            raise ValueError("Archive contains no images/ or labels/ entries.")

        for member, destination in members:
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue

            if destination.exists():
                if not destination.is_file():
                    raise ValueError(f"Expected a file at: {destination}")
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f"{destination.name}.download-part"
            )

            try:
                with (
                    archive.open(member) as source,
                    temporary.open("wb") as target,
                ):
                    shutil.copyfileobj(source, target)

                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)


def _download_from_import_source(target: Path) -> None:
    from omegaconf import OmegaConf

    metadata = OmegaConf.load(target)
    dependencies = metadata.get("deps", [])
    outputs = metadata.get("outs", [])

    if len(dependencies) != 1 or len(outputs) != 1:
        raise ValueError(
            "Expected one source and one output in dataset DVC metadata."
        )

    source_url = str(dependencies[0].get("path", ""))
    expected_md5 = str(outputs[0].get("md5", "")).lower()

    if not source_url.startswith(("https://", "http://")):
        raise ValueError("Expected an HTTP(S) dataset import source.")

    if len(expected_md5) != 32 or any(
        character not in "0123456789abcdef" for character in expected_md5
    ):
        raise ValueError("Expected a file MD5 in dataset DVC metadata.")

    destination = config.DATA_ARCHIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".dataset-download-",
        dir=destination.parent,
    ) as directory:
        temporary = Path(directory) / destination.name

        subprocess.run(
            [
                config.DVC_EXECUTABLE,
                "get-url",
                source_url,
                str(temporary),
            ],
            cwd=config.PROJECT_ROOT,
            check=True,
        )

        if not temporary.is_file():
            raise FileNotFoundError(
                f"DVC did not download the dataset: {temporary}"
            )

        digest = hashlib.md5()

        with temporary.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        actual_md5 = digest.hexdigest()

        if actual_md5 != expected_md5:
            raise ValueError(
                "Downloaded dataset checksum mismatch.\n"
                f"Expected: {expected_md5}\n"
                f"Actual:   {actual_md5}"
            )

        if destination.exists():
            raise FileExistsError(
                f"Archive appeared during download: {destination}"
            )

        temporary.replace(destination)

    print("Dataset downloaded from its DVC import source; MD5 verified.")


def download_data() -> Path:
    target = Path(config.DVC_TARGET)

    if not target.is_absolute():
        target = config.PROJECT_ROOT / target

    if not target.is_file():
        raise FileNotFoundError(f"DVC metadata not found: {target}")

    archive_path = config.DATA_ARCHIVE_PATH

    if archive_path.exists() and not archive_path.is_file():
        raise ValueError(f"Expected an archive file at: {archive_path}")

    if not archive_path.is_file():
        print("Downloading dataset from its DVC import source.")
        _download_from_import_source(target)
    else:
        print(f"Using existing dataset archive: {archive_path}")

    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    _extract_dataset(
        archive_path,
        config.DATA_ROOT,
    )

    required = (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    )

    for relative in required:
        directory = config.DATA_ROOT / relative

        if not directory.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found after extraction: {directory}"
            )

        if not any(directory.iterdir()):
            raise ValueError(f"Dataset directory is empty: {directory}")

    print(f"Dataset is available at: {config.DATA_ROOT}")
    return config.DATA_ROOT
