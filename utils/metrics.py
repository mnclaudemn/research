
def evaluate(model, loader, criterion, device):

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            preds = model(x)
            loss = criterion(preds, y)

            total_loss += loss.item()

            _, predicted = preds.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

    acc = 100. * correct / total
    return total_loss / len(loader), acc
