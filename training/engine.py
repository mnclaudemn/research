
import torch

def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        preds = model(x)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
