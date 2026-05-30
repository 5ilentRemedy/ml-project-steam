from pathlib import Path

from steam_ml.core import DATA_DIR, latest_raw_csv, run_step


def main() -> dict[str, str]:
    source = latest_raw_csv()
    return {
        "status": "local_raw_dataset_ready",
        "source": str(source),
        "hint": f"Umiesc nowy plik CSV w {DATA_DIR} jako games_YYYYMMDD_HHMMSS.csv, aby pipeline uzyl go automatycznie.",
    }


if __name__ == "__main__":
    run_step("01 data collection", main)

