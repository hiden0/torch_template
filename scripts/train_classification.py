import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor
import torch.nn as nn
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.custom_CNN import CustomCNN
from data.PetImages_dataset import get_dataloader
from utils.metrics import compute_metrics


###############CONFIG PARAMETERS###############

CSV_PATH = "/srv/hdd2/javber/dataset.csv"
NUM_CLASSES = 2
BATCH_SIZE = 128
NUM_WORKERS = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
OPTIMIZER = "adam"
WEIGHT_DECAY = 0
IMAGE_SIZE = 256
EXPERIMENT_NAME = "custom_CNN_1"
FROM_CHECKPOINT = False
CHECKPOINT_PATH = None
LOSS_FN = "cross_entropy"


###############################################


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
        print(f"Invalid loss type: {loss_type}. Using CrossEntropyLoss.")
        return nn.CrossEntropyLoss()  # Por defecto, clasificación multiclase


def plot_confusion_matrix(cm, class_names):
    """Genera una imagen de la matriz de confusión con Seaborn y la convierte en tensor."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    fig.canvas.draw()

    # Convertir la imagen en un array de numpy (H, W, C)
    img_array = np.array(fig.canvas.renderer.buffer_rgba())[
        :, :, :3
    ]  # Eliminar canal alfa (RGBA → RGB)
    plt.close(fig)  # Cerrar la figura para liberar memoria

    # Convertir numpy array en tensor (H, W, C)
    img_tensor = (
        torch.from_numpy(img_array).permute(0, 1, 2).float() / 255.0
    )  # Normalizar valores a [0,1]
    return img_array


# Buscar el device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Crear directorios para guardar checkpoints
checkpoint_dir = f"../checkpoints/{EXPERIMENT_NAME}"
os.makedirs(checkpoint_dir, exist_ok=True)

# Inicializar TensorBoard
log_dir = f"../runs/training_logs/{EXPERIMENT_NAME}"
writer = SummaryWriter(log_dir=log_dir)

# Crear DataLoaders para cada split
train_loader = get_dataloader(
    CSV_PATH,
    split="train",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    img_size=IMAGE_SIZE,
)
val_loader = get_dataloader(
    CSV_PATH,
    split="val",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    img_size=IMAGE_SIZE,
)

model = CustomCNN(
    input_channels=3,
    num_classes=NUM_CLASSES,
    input_size=(IMAGE_SIZE, IMAGE_SIZE),
    conv_layers=[(32, 3), (64, 3), (128, 3)],  # 3 capas convolucionales
    dropout_rate=0.5,
    use_batchnorm=True,
    pooling_type="avg",
    dense_neurons=256,
).to(device)

if FROM_CHECKPOINT:
    assert CHECKPOINT_PATH is not None, "Debes proporcionar la ruta al checkpoint"
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

optimizer = select_optimizer(model, OPTIMIZER, LEARNING_RATE, WEIGHT_DECAY)
criterion = get_loss_function(LOSS_FN)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calcular accuracy en el batch
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        batch_accuracy = correct / total if total > 0 else 0.0

        # Imprimir progreso cada N batches
        if (batch_idx + 1) % 10 == 0:  # Muestra la métrica cada 10 batches

            # Sobrescribir la línea en la terminal
            sys.stdout.write(
                f"\rEpoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                f"Train Loss: {running_loss / (batch_idx+1):.4f}, "
                f"Train Acc: {batch_accuracy:.4f}"
            )
            sys.stdout.flush()

    # Calcular métricas en entrenamiento y validación
    train_metrics = compute_metrics(model, train_loader, criterion, device)
    val_metrics = compute_metrics(model, val_loader, criterion, device)

    # Añadir matriz de confusión a TensorBoard
    train_cm_image = plot_confusion_matrix(
        train_metrics["confusion_matrix"],
        class_names=[str(i) for i in range(NUM_CLASSES)],
    )

    val_cm_image = plot_confusion_matrix(
        val_metrics["confusion_matrix"],
        class_names=[str(i) for i in range(NUM_CLASSES)],
    )

    # Guardar métricas en TensorBoard
    writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
    writer.add_scalar("Loss/Validation", val_metrics["loss"], epoch)
    writer.add_scalar("Accuracy/Train", train_metrics["accuracy"], epoch)
    writer.add_scalar("Accuracy/Validation", val_metrics["accuracy"], epoch)
    writer.add_scalar("Precision/Train", train_metrics["precision"], epoch)
    writer.add_scalar("Precision/Validation", val_metrics["precision"], epoch)
    writer.add_scalar("Recall/Train", train_metrics["recall"], epoch)
    writer.add_scalar("Recall/Validation", val_metrics["recall"], epoch)
    writer.add_scalar("F1-score/Train", train_metrics["f1_score"], epoch)
    writer.add_scalar("F1-score/Validation", val_metrics["f1_score"], epoch)
    writer.add_image("Confusion_Matrix/Train", train_cm_image, epoch, dataformats="HWC")
    writer.add_image(
        "Confusion_Matrix/Validation", val_cm_image, epoch, dataformats="HWC"
    )

    print(
        f"\nEpoch {epoch+1}: "
        f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
        f"Train Precision: {train_metrics['precision']:.4f}, Train Recall: {train_metrics['recall']:.4f}, "
        f"Train F1: {train_metrics['f1_score']:.4f} "
    )
    print(
        f"          Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
        f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}, "
        f"Val F1: {val_metrics['f1_score']:.4f} "
    )

    # Guardar checkpoint de la época
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        checkpoint_path,
    )

# Cerrar TensorBoard
writer.close()
