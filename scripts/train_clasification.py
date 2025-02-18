# scripts/train.py


# import torch
import torch.optim as optim
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.custom_model import CustomModel
from data.example_dataset import get_dataloader
from utils.metrics import compute_accuracy, compute_loss

###############CONFIG PARAMETERS###############

CSV_PATH = "/srv/hdd2/javber/dataset.csv"
NUM_CLASSES = 2
BATCH_SIZE = 64
NUM_WORKERS = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
OPTIMIZER = "adam"
WEIGHT_DECAY = 0
###############################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear DataLoaders para cada split
train_loader = get_dataloader(
    CSV_PATH, split="train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)
val_loader = get_dataloader(
    CSV_PATH, split="val", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)


model = CustomModel(num_classes=NUM_CLASSES).to(device)
optimizer = select_optimizer(model, OPTIMIZER, LEARNING_RATE, WEIGHT_DECAY)


criterion = nn.CrossEntropyLoss()

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calcular métricas
    train_loss = running_loss / len(train_loader)
    train_accuracy = compute_accuracy(model, train_loader, device)
    val_loss = compute_loss(model, val_loader, criterion, device)
    val_accuracy = compute_accuracy(model, val_loader, device)

    print(
        f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
    )


def select_optimizer(model, optimizer_name, learning_rate=0.001, w_decay=0):
    """
    Selecciona y devuelve el optimizador adecuado para el modelo dado.

    Parámetros:
    - model (torch.nn.Module): El modelo para el cual se va a utilizar el optimizador.
    - optimizer_name (str): El nombre del optimizador a utilizar. Opciones:
        - "adam": Adam optimizer.
        - "adamw": AdamW optimizer.
        - "sgd": Stochastic Gradient Descent (SGD) optimizer.
        - "rmsprop": RMSprop optimizer.
        - "adagrad": Adagrad optimizer.
        - "adadelta": Adadelta optimizer.
    - learning_rate (float, opcional): La tasa de aprendizaje para el optimizador. Por defecto es 0.001.
    - w_decay (float, opcional): El factor de decaimiento de peso (weight decay) para el optimizador. Por defecto es 0.

    Retorna:
    - torch.optim.Optimizer: Una instancia del optimizador seleccionado.

    Lanza:
    - ValueError: Si el nombre del optimizador no es válido.
    """
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=w_decay
        )
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            alpha=0.99,
            eps=1e-8,
            weight_decay=w_decay,
            momentum=0.9,
        )
    elif optimizer_name == "adagrad":
        return optim.Adagrad(
            model.parameters(), lr=learning_rate, lr_decay=0, weight_decay=w_decay
        )
    elif optimizer_name == "adadelta":
        return optim.Adadelta(model.parameters(), rho=0.9, eps=1e-6)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")


import torch.nn as nn


def get_loss_function(loss_type="cross_entropy"):
    """
    Devuelve la mejor función de pérdida para clasificación en redes convolucionales.

    Parámetros:
    - loss_type (str): Tipo de pérdida. Opciones:
        - "cross_entropy": Para clasificación multiclase.
        - "bce": Para clasificación binaria con BCEWithLogitsLoss.
        - "focal": Para clasificación en datasets desbalanceados.

    Retorna:
    - Instancia de la función de pérdida en PyTorch.
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "focal":
        from torch.nn import functional as F

        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2, reduction="mean"):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction

            def forward(self, inputs, targets):
                BCE_loss = F.binary_cross_entropy_with_logits(
                    inputs, targets, reduction="none"
                )
                pt = torch.exp(-BCE_loss)
                F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
                return F_loss.mean() if self.reduction == "mean" else F_loss.sum()

        return FocalLoss()

    else:
        return nn.CrossEntropyLoss()  # Por defecto, clasificación multiclase
