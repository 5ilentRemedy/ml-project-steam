from steam_ml.core import AdvancedModelEvaluator, run_step


if __name__ == "__main__":
    run_step("08 model evaluation", lambda: AdvancedModelEvaluator().run())

