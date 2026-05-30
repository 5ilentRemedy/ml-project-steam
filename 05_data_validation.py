from steam_ml.core import DataValidator, run_step


if __name__ == "__main__":
    run_step("05 data validation", lambda: DataValidator().run())

