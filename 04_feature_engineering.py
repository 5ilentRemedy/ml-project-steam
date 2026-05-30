from steam_ml.core import FeatureEngineer, run_step


if __name__ == "__main__":
    run_step("04 feature engineering", lambda: FeatureEngineer().run())

