import torch

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):

    model.train()
    total_loss = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed Precision
            with torch.amp.autocast("cuda"):
                preds = model(x)
                loss = criterion(preds, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            # CPU or no AMP
            preds = model(x)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
