import torch
import torch.nn as nn
from datasets.medical_dataset import get_loaders
from models.backbones import get_model
from training.engine import train_one_epoch, evaluate
from utils.seed import set_seed

def main():

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, num_classes = get_loaders(
        "/content/train",
        "/content/test",
        batch_size=16
    )

    model = get_model("resnet50", num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):

        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch} | loss {loss:.4f} | acc {acc:.2f}")

if __name__ == "__main__":
    main()
