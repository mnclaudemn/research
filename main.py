import torch
import torch.nn as nn

from utils.config import load_config
from utils.seed import set_seed
from utils.dataset_report import analyze_dataset, show_samples
from datasets.medical_dataset import get_loaders
from models.backbones import get_model
from utils.fine_tuning import unfreeze_last_n
from training.engine import train_one_epoch, evaluate
from utils.experiment_logger import ExperimentLogger


def main():

    # ==========================
    # Load Config
    # ==========================
    config = load_config()

    dataset_root = config["dataset_root"]
    model_name = config["model"]["name"]
    n_unfreeze = config["model"]["n_unfreeze"]
    lr = config["model"]["lr"]
    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epochs"]
    image_size = config["image_size"]
    seed = config["seed"]

    # ==========================
    # Setup
    # ==========================
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================
    # Dataset analysis
    # ==========================
    splits, train_ds = analyze_dataset(dataset_root)
    show_samples(train_ds, num_classes=5)

    # ==========================
    # Data loaders
    # ==========================
    train_loader, val_loader, test_loader, num_classes = get_loaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        image_size=image_size
    )

    # ==========================
    # Model
    # ==========================
    model = get_model(model_name, num_classes).to(device)

    trainable_blocks = unfreeze_last_n(
        model=model,
        model_name=model_name,
        n=n_unfreeze
    )

    print(f"\nModel: {model_name}")
    print(f"Trainable blocks: {trainable_blocks}\n")

    # ==========================
    # Loss + Optimizer
    # ==========================
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # ==========================
    # Logger (CREATE EARLY)
    # ==========================
    logger = ExperimentLogger()

    best_acc = 0
    best_metrics = None

    # ==========================
    # Training loop
    # ==========================
    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        eval_loader = val_loader if val_loader is not None else test_loader

        val_loss, metrics = evaluate(
            model, eval_loader, criterion, device
        )

        acc = metrics["accuracy"]

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.2f}"
        )

        # track best
        if acc > best_acc:
            best_acc = acc
            best_metrics = metrics

    # ==========================
    # Save model
    # ==========================
    model_path = f"checkpoints/{model_name}.pt"
    torch.save(model.state_dict(), model_path)

    # ==========================
    # LOG EXPERIMENT (IMPORTANT)
    # ==========================
    logger.log(
        config={
            "dataset_root": dataset_root,
            "model": model_name,
            "n_unfreeze": n_unfreeze,
            "image_size": image_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "optimizer": "adam"
        },
        metrics=best_metrics,
        loss=val_loss,
        model_path=model_path
    )


if __name__ == "__main__":
    main()
