import torch
import torch.nn as nn

from datasets.medical_dataset import get_loaders
from models.backbones import get_model
from training.engine import train_one_epoch, evaluate
from utils.seed import set_seed
from utils.fine_tuning import unfreeze_last_n


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
    # Data
    # ==========================
    train_loader, test_loader, num_classes = get_loaders(
        "/content/train",
        "/content/test",
        batch_size=16
    )

    # ==========================
    # Model Configuration
    # ==========================
    model_name = "resnet50"
    n_unfreeze = 2

    # Build model
    model = get_model(
        model_name,
        num_classes
    ).to(device)

    # Freeze everything and unfreeze last n blocks
    trainable_blocks = unfreeze_last_n(
        model=model,
        model_name=model_name,
        n=n_unfreeze
    )

    print(f"\nModel: {model_name}")
    print(f"Trainable blocks: {trainable_blocks}\n")

    # ==========================
    # Loss and Optimizer
    # ==========================
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
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

        val_loss, acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.2f}%"
        )


if __name__ == "__main__":
    main()
