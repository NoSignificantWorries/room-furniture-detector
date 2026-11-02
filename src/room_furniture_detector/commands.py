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


def main() -> None:
    fire.Fire(
        {
            "analyze": analyze,
            "train": train,
        }
    )


if __name__ == "__main__":
    main()
