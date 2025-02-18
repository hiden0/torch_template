import torch


def compute_accuracy(model, dataloader, device):
    """Calcula la accuracy en un dataloader dado."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def compute_loss(model, dataloader, criterion, device):
    """Calcula la loss promedio en un dataloader dado."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
