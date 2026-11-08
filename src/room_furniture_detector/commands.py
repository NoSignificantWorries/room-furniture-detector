import fire

from room_furniture_detector import config


def download_data() -> None:
    from room_furniture_detector.download import (
        download_data as download_dataset,
    )

    download_dataset()


def analyze(
    data_dir: str | None = None,
    output_dir: str | None = None,
) -> None:
    from room_furniture_detector.analysis import analyze_dataset

    analyze_dataset(
        data_dir=(
            str(config.ANALYZE_DATA_DIR) if data_dir is None else data_dir
        ),
        output_dir=(
            str(config.ANALYZE_OUTPUT_DIR) if output_dir is None else output_dir
        ),
    )


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
    split: str = "test",
) -> None:
    from room_furniture_detector.evaluation import test_model

    test_model(
        overrides=overrides,
        dry_run=dry_run,
        source_run_id=source_run_id,
        split=split,
    )


def prepare_data(overrides: str = "") -> None:
    from room_furniture_detector.data_split import prepare_splits
    from room_furniture_detector.training import load_config

    cfg = load_config(overrides)

    prepare_splits(cfg, config.PROJECT_ROOT)


def predict(
    image: str,
    weights: str | None = None,
    output: str | None = None,
    conf: float | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    save_image: bool | None = None,
) -> None:
    from room_furniture_detector.prediction import predict_image

    predict_image(
        image=image,
        weights=(str(config.MODEL_PT_PATH) if weights is None else weights),
        output=(str(config.PREDICT_OUTPUT_PATH) if output is None else output),
        conf=(config.PREDICT_CONF if conf is None else conf),
        imgsz=(config.PREDICT_IMGSZ if imgsz is None else imgsz),
        device=(config.PREDICT_DEVICE if device is None else str(device)),
        save_image=(
            config.PREDICT_SAVE_IMAGE if save_image is None else save_image
        ),
    )


def main() -> None:
    fire.Fire(
        {
            "download_data": download_data,
            "analyze": analyze,
            "prepare_data": prepare_data,
            "train": train,
            "test": test,
            "predict": predict,
        }
    )


if __name__ == "__main__":
    main()
