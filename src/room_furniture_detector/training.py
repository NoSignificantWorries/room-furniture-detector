import os
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def load_config(overrides: str = "") -> DictConfig:
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "configs"

    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
    ):
        config = compose(
            config_name="config",
            overrides=overrides.split(),
        )

    return config


def get_git_metadata(project_root: Path) -> dict[str, str]:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    return {
        "git_commit": commit_result.stdout.strip(),
        "git_dirty": str(bool(status_result.stdout.strip())).lower(),
    }


def prepare_dataset_yaml(
    config: DictConfig,
    project_root: Path,
    output_dir: Path,
) -> Path:
    data_root = (project_root / config.data.root).resolve()

    required_dirs = (
        data_root / config.data.train,
        data_root / config.data.val,
        data_root / "labels" / "train",
        data_root / "labels" / "val",
    )

    for directory in required_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {directory}"
            )

    if len(config.data.names) != config.data.num_classes:
        raise ValueError("num_classes does not match the number of names.")

    dataset_config = OmegaConf.create(
        {
            "path": data_root.as_posix(),
            "train": config.data.train,
            "val": config.data.val,
            "names": OmegaConf.to_container(config.data.names),
        }
    )

    dataset_yaml = output_dir / "dataset.yaml"
    OmegaConf.save(config=dataset_config, f=dataset_yaml)

    return dataset_yaml


def train_model(overrides: str = "", dry_run: bool = False) -> None:
    config = load_config(overrides)
    print(OmegaConf.to_yaml(config, resolve=True))

    if dry_run:
        return

    import mlflow
    from ultralytics import YOLO, settings

    project_root = Path(__file__).resolve().parents[2]
    output_dir = (project_root / config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = get_git_metadata(project_root)
    dataset_yaml = prepare_dataset_yaml(
        config,
        project_root,
        output_dir,
    )

    os.environ["MLFLOW_TRACKING_URI"] = config.logging.tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = config.logging.experiment_name
    os.environ["MLFLOW_RUN"] = config.run_name

    mlflow.set_tracking_uri(config.logging.tracking_uri)
    settings.update({"mlflow": True})

    model = YOLO(config.model.weights)

    def log_project_metadata(trainer) -> None:
        if mlflow.active_run() is None:
            raise RuntimeError(
                "No active MLflow run. Check the Ultralytics MLflow integration."
            )

        run_dir = Path(trainer.save_dir)
        config_path = run_dir / "resolved_config.yaml"

        OmegaConf.save(
            config=OmegaConf.create(
                OmegaConf.to_container(config, resolve=True)
            ),
            f=config_path,
        )

        mlflow.set_tags(metadata)
        mlflow.log_artifact(str(config_path), artifact_path="configuration")
        mlflow.log_artifact(str(dataset_yaml), artifact_path="configuration")

    model.add_callback("on_train_start", log_project_metadata)

    training_args = OmegaConf.to_container(
        config.training,
        resolve=True,
    )
    augmentation_args = OmegaConf.to_container(
        config.augmentation,
        resolve=True,
    )

    model.train(
        data=str(dataset_yaml),
        project=str(output_dir),
        name=config.run_name,
        **training_args,
        **augmentation_args,
    )
