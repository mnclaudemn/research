import torch
import torch.nn as nn

from datasets.medical_dataset import get_loaders
from models.backbones import get_model
from training.engine import train_one_epoch, evaluate
from utils.seed import set_seed
from utils.fine_tuning import unfreeze_last_n
from utils.dataset_report import analyze_dataset, show_samples


def main():

    # ==========================
    # Reproducibility
    # ==========================
    set_seed(42)

    # ==========================
    # Device
    # ==========================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================
    # Dataset Analysis (NEW)
    # ==========================
    dataset_root = "/content/dataset"

    splits, train_ds = analyze_dataset(dataset_path)
    show_samples(train_ds, num_classes=5)

    # ==========================
    # Data Loaders (FIXED)
    # ==========================
    train_loader, val_loader, test_loader, num_classes = get_loaders(
        dataset_root=dataset_root,
        batch_size=16
    )

    # ==========================
    # Model Configuration
    # ==========================
    model_name = "resnet50"
    n_unfreeze = 2

    model = get_model(
        model_name,
        num_classes
    ).to(device)

    # ==========================
    # Fine-tuning
    # ==========================
    trainable_blocks = unfreeze_last_n(
        model=model,
        model_name=model_name,
        n=n_unfreeze
    )

    print(f"\nModel: {model_name}")
    print(f"Trainable blocks: {trainable_blocks}\n")

    # ==========================
    # Loss & Optimizer
    # ==========================
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    # ==========================
    # Training Loop
    # ==========================
    epochs = 10

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        # use val if exists, otherwise test
        eval_loader = val_loader if val_loader is not None else test_loader

        val_loss, acc = evaluate(
            model,
            eval_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.2f}%"
        )


if __name__ == "__main__":
    main()
