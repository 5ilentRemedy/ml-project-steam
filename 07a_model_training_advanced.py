from steam_ml.core import AdvancedModelTrainer, run_step


if __name__ == "__main__":
    run_step("07a advanced model training", lambda: AdvancedModelTrainer().run())

