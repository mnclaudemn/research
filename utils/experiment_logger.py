import pandas as pd
import os
from datetime import datetime


class ExperimentLogger:

    def __init__(self, file_path="experiments.xlsx"):
        self.file_path = file_path

        if os.path.exists(file_path):
            self.df = pd.read_excel(file_path)
        else:
            self.df = pd.DataFrame()

    # ==========================
    # Create unique experiment name
    # ==========================
    def create_name(self, config):

        time = datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"{config['model']}_{config['dataset']}_{time}"

    # ==========================
    # Add experiment row
    # ==========================
    def log(self, config, result, model_path):

        exp_name = self.create_name(config)

        row = {
            "experiment_name": exp_name,

            # dataset info
            "dataset": config["dataset"],
            "image_size": config["image_size"],

            # model info
            "model": config["model"],
            "n_unfreeze": config["n_unfreeze"],

            # training info
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "lr": config["lr"],
            "optimizer": config["optimizer"],

            # results
            "best_accuracy": result["accuracy"],
            "val_loss": result["val_loss"],

            # outputs
            "model_path": model_path
        }

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

        self.save()

        print(f"[LOGGED] {exp_name}")

        return exp_name

    # ==========================
    # Save Excel file
    # ==========================
    def save(self):
        self.df.to_excel(self.file_path, index=False)
