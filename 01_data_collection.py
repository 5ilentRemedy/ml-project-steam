import os

from steam_ml.core import DEFAULT_KAGGLE_DATASET, download_kaggle_dataset, run_step


def main() -> dict[str, str]:
    dataset = os.environ.get("KAGGLE_DATASET", DEFAULT_KAGGLE_DATASET)
    source = download_kaggle_dataset(dataset)
    return {
        "status": "downloaded",
        "dataset": dataset,
        "source": str(source),
    }


if __name__ == "__main__":
    run_step("01 data collection", main)
