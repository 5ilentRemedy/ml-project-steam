from steam_ml.core import DataExporter, run_step


if __name__ == "__main__":
    run_step("06 data export", lambda: DataExporter().run())

