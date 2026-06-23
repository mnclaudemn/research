import torch

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        # If device is a string, .to() handles it perfectly
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) # Optimization: cleaner than zero_grad()

        if scaler is not None:
            # Explicitly setting dtype is the modern PyTorch standard
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                preds = model(x)
                loss = criterion(preds, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Clean check whether we are using a CUDA device (handles string or torch.device)
    is_cuda = (device == "cuda" or (hasattr(device, "type") and device.type == "cuda"))

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Safely enable autocast only if CUDA is active
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                preds = model(x)
                loss = criterion(preds, y)

            total_loss += loss.item()
            correct += (preds.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    acc = correct / total
    return total_loss / len(loader), acc
