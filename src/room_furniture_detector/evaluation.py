import hashlib
import subprocess
from contextlib import nullcontext
from pathlib import Path

import mlflow
from omegaconf import OmegaConf
from ultralytics import YOLO

from room_furniture_detector import config
from room_furniture_detector.training import load_config


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _git_tags(project_root: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "git_commit": git("rev-parse", "HEAD"),
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": str(bool(git("status", "--porcelain"))).lower(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_metadata": "unavailable"}


def test_model(
    overrides: str = "",
    dry_run: bool = False,
    source_run_id: str = "",
    split: str = "test",
) -> None:
    if split not in {"val", "test"}:
        raise ValueError("split must be either 'val' or 'test'.")

    cfg = load_config(overrides)

    evaluation = cfg.get("evaluation", {})

    evaluation_conf = float(evaluation.get("conf", config.EVALUATION_CONF))
    evaluation_iou = float(evaluation.get("iou", config.EVALUATION_IOU))
    evaluation_save_json = bool(
        evaluation.get("save_json", config.EVALUATION_SAVE_JSON)
    )
    evaluation_plots = bool(evaluation.get("plots", config.EVALUATION_PLOTS))

    if not 0.0 <= evaluation_conf <= 1.0:
        raise ValueError("Evaluation confidence must be between 0 and 1.")

    if not 0.0 <= evaluation_iou <= 1.0:
        raise ValueError("Evaluation IoU must be between 0 and 1.")

    project_root = config.PROJECT_ROOT

    weights = _resolve_path(str(cfg.model.weights), project_root)
    output_dir = _resolve_path(str(cfg.output_dir), project_root)
    dataset_yaml = output_dir / "dataset.yaml"

    if dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print(f"Weights: {weights}")
        print(f"Dataset: {dataset_yaml}")
        print(f"Output directory: {output_dir}")
        print(f"Evaluation split: {split}")
        print(f"Source training run: {source_run_id or 'not specified'}")
        return

    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    dataset_cfg = OmegaConf.load(dataset_yaml)
    if not dataset_cfg.get(split):
        raise ValueError(f"Split '{split}' is missing in {dataset_yaml}")

    checkpoint_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    tracking_enabled = bool(cfg.logging.tracking_uri)

    if tracking_enabled:
        mlflow.set_tracking_uri(str(cfg.logging.tracking_uri))
        mlflow.set_experiment(str(cfg.logging.experiment_name))

    run_context = (
        mlflow.start_run(run_name=str(cfg.run_name))
        if tracking_enabled
        else nullcontext()
    )

    with run_context:
        if tracking_enabled:
            tags = {
                **_git_tags(project_root),
                "job_type": "evaluation",
                "evaluation_split": split,
                "checkpoint_sha256": checkpoint_sha256,
            }
            if source_run_id:
                tags["source_training_run_id"] = source_run_id

            mlflow.set_tags(tags)
            mlflow.log_params(
                {
                    "model.weights": str(weights),
                    "data.yaml": str(dataset_yaml),
                    "data.split": split,
                    "evaluation.imgsz": cfg.training.imgsz,
                    "evaluation.batch": cfg.training.batch,
                    "evaluation.device": str(cfg.training.device),
                    "evaluation.workers": cfg.training.workers,
                    "evaluation.conf": evaluation_conf,
                    "evaluation.nms_iou": evaluation_iou,
                    "evaluation.save_json": evaluation_save_json,
                    "evaluation.plots": evaluation_plots,
                }
            )
            mlflow.log_text(
                OmegaConf.to_yaml(cfg, resolve=True),
                "configuration/config.yaml",
            )
            mlflow.log_text(
                overrides,
                "configuration/overrides.txt",
            )
            mlflow.log_artifact(
                str(dataset_yaml),
                artifact_path="configuration",
            )

        model = YOLO(str(weights))
        validation_output: dict[str, Path] = {}

        def capture_save_dir(validator) -> None:
            validation_output["save_dir"] = Path(validator.save_dir)

        model.add_callback("on_val_end", capture_save_dir)

        results = model.val(
            data=str(dataset_yaml),
            split=split,
            imgsz=cfg.training.imgsz,
            batch=cfg.training.batch,
            device=cfg.training.device,
            workers=cfg.training.workers,
            conf=evaluation_conf,
            iou=evaluation_iou,
            plots=evaluation_plots,
            save_json=evaluation_save_json,
            project=str(output_dir),
            name=str(cfg.run_name),
            exist_ok=False,
        )

        metrics = results.results_dict
        evaluation_metrics = {
            f"{split}/precision": float(metrics["metrics/precision(B)"]),
            f"{split}/recall": float(metrics["metrics/recall(B)"]),
            f"{split}/mAP50": float(metrics["metrics/mAP50(B)"]),
            f"{split}/mAP50-95": float(metrics["metrics/mAP50-95(B)"]),
        }

        save_dir = validation_output["save_dir"]
        print(f"Evaluation split: {split}")
        print(f"Evaluation metrics: {evaluation_metrics}")
        print(f"Results directory: {save_dir}")

        if tracking_enabled:
            mlflow.log_metrics(evaluation_metrics)
            mlflow.log_artifacts(
                str(save_dir),
                artifact_path="evaluation",
            )
