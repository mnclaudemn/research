import torch
import torch.nn as nn

from utils.config import load_config
from utils.seed import set_seed
from utils.dataset_report import analyze_dataset
from datasets.medical_dataset import get_loaders
from models.backbones import get_model
from utils.fine_tuning import unfreeze_last_n
from training.engine import train_one_epoch, evaluate
from utils.logger import save_result

import yaml


def run_experiment(config, dataset_root):

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, num_classes = get_loaders(
        dataset_root=dataset_root,
        batch_size=16
    )

    model = get_model(config["model"], num_classes).to(device)

    trainable_blocks = unfreeze_last_n(
        model,
        config["model"],
        config["n_unfreeze"]
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    epochs = 5  # keep small for experiments

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        eval_loader = val_loader if val_loader else test_loader

        val_loss, acc = evaluate(
            model, eval_loader, criterion, device
        )

    return acc, val_loss


def main():

    # load dataset config (optional)
    dataset_root = "/content/dataset"

    # load experiments
    with open("configs/experiments.yaml", "r") as f:
        experiments = yaml.safe_load(f)["experiments"]

    results_file = "results.csv"

    for exp in experiments:

        print("\n====================================")
        print(f"Running: {exp}")
        print("====================================\n")

        acc, val_loss = run_experiment(exp, dataset_root)

        result = {
            "model": exp["model"],
            "n_unfreeze": exp["n_unfreeze"],
            "accuracy": acc,
            "val_loss": val_loss
        }

        save_result(results_file, result)

        print(f"Saved: {result}")


if __name__ == "__main__":
    main()
