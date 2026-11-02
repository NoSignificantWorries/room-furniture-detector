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


def prepare_data(overrides: str = "") -> None:
    from pathlib import Path

    from room_furniture_detector.data_split import prepare_splits
    from room_furniture_detector.training import load_config

    project_root = Path(__file__).resolve().parents[2]
    config = load_config(overrides)

    prepare_splits(config, project_root)


def main() -> None:
    fire.Fire(
        {
            "analyze": analyze,
            "prepare_data": prepare_data,
            "train": train,
        }
    )


if __name__ == "__main__":
    main()
