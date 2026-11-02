import fire

from room_furniture_detector.analysis import analyze_dataset


def analyze(
    data_dir: str = "data",
    output_dir: str = "reports/dataset",
) -> None:
    analyze_dataset(data_dir=data_dir, output_dir=output_dir)


def train(
    overrides: str = "",
    dry_run: bool = False,
) -> None:
    from room_furniture_detector.training import train_model

    train_model(
        overrides=overrides,
        dry_run=dry_run,
    )


def test(
    overrides: str = "",
    dry_run: bool = False,
    source_run_id: str = "",
) -> None:
    from room_furniture_detector.evaluation import test_model

    test_model(
        overrides=overrides,
        dry_run=dry_run,
        source_run_id=source_run_id,
    )


def prepare_data(overrides: str = "") -> None:
    from pathlib import Path

    from room_furniture_detector.data_split import prepare_splits
    from room_furniture_detector.training import load_config

    project_root = Path(__file__).resolve().parents[2]
    config = load_config(overrides)

    prepare_splits(config, project_root)


def predict(
    image: str,
    weights: str = "outputs/yolo11n_baseline/weights/best.pt",
    output: str = "outputs/predictions/room.json",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
    save_image: bool = False,
) -> None:
    from room_furniture_detector.prediction import predict_image

    predict_image(
        image=image,
        weights=weights,
        output=output,
        conf=conf,
        imgsz=imgsz,
        device=device,
        save_image=save_image,
    )


def main() -> None:
    fire.Fire(
        {
            "analyze": analyze,
            "prepare_data": prepare_data,
            "train": train,
            "test": test,
            "predict": predict,
        }
    )


if __name__ == "__main__":
    main()
