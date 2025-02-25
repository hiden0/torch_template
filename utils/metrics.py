import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(model, dataloader, criterion, device, num_classes=2):
    """Calcula múltiples métricas en un dataloader dado."""

    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(
                outputs, 1
            )  # Predicción de la clase con mayor probabilidad
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Accuracy
    accuracy = correct / total if total > 0 else 0.0
    # Precisión, Recall y F1-Score
    precision = precision_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    # Matriz de Confusión
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    # Pérdida Promedio
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "loss": avg_loss,
        "confusion_matrix": conf_matrix,
    }
