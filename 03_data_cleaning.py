from steam_ml.core import DataCleaner, run_step


if __name__ == "__main__":
    run_step("03 data cleaning", lambda: DataCleaner().run())

