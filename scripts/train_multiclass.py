import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import sys
import os
import torch.nn as nn
import shutil

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_dir, ".."))

from models.custom_CNN import CustomCNN
from data.PetImages_dataset import get_dataloader
from engine.metrics import compute_metrics, plot_confusion_matrix
from engine.training_config import select_optimizer

###############################################################################################################################
###################################################### CONFIG PARAMETERS ######################################################
###############################################################################################################################

### GENERAL ###
CSV_PATH = "/srv/hdd2/javber/dataset.csv"
NUM_CLASSES = 2
EXPERIMENT_NAME = "custom_CNN_3"
FROM_CHECKPOINT = False
CHECKPOINT_PATH = False
NUM_WORKERS = 8

### TRAIN HPARAMS ###
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SCHEDULER = "step"  # "step" o "plateau"
SCHEDULER_STEP = 5
SCHEDULER_REDUCTION = 0.9
NUM_EPOCHS = 80
OPTIMIZER = "adamw"
WEIGHT_DECAY = 0.0001
IMAGE_SIZE = 256


### DATA AUGMENTATION ###
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CROP_SCALE = (0.8, 1.0)  # Rango de escala para el recorte aleatorio
ROTATION_DEGREES = 15  # Rotación aleatoria en grados
HORIZONTAL_FLIP_PROB = 0.5  # Probabilidad de aplicar flip horizontal
BRIGHTNESS = 0.2
CONTRAST = 0.2
SATURATION = 0.2
HUE = 0.1

###CNN CONFIG###
CNN_CONV_LAYERS = [(32, 3), (64, 3), (128, 3), (256, 3)]
CNN_DROPOUT_RATE = 0.5
CNN_USE_BATCHNORM = True
CNN_POOLING_TYPE = "avg"
CNN_DENSE_NEURONS = 256

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# TENSORBOARD AND CHECKPOINTS
log_dir = os.path.join(base_dir, f"../runs/training_logs/{EXPERIMENT_NAME}")
checkpoint_dir = os.path.join(base_dir, f"../runs/checkpoints/{EXPERIMENT_NAME}")

if os.path.exists(checkpoint_dir) and not FROM_CHECKPOINT:
    response = (
        input(
            f"Experiment '{EXPERIMENT_NAME}' already exists. Do you want to overwrite it? (y/n): "
        )
        .strip()
        .lower()
    )

    if response != "y":
        print("Exiting script without overwriting experiment.")
        sys.exit()
    else:
        print(f"The old experiment '{EXPERIMENT_NAME}' will be overwritten.")
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree(log_dir, ignore_errors=True)

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Guardar hiperparámetros en TensorBoard
hparams = {
    "CSV_PATH": CSV_PATH,
    "NUM_CLASSES": NUM_CLASSES,
    "EXPERIMENT_NAME": EXPERIMENT_NAME,
    "FROM_CHECKPOINT": FROM_CHECKPOINT,
    "CHECKPOINT_PATH": CHECKPOINT_PATH,
    "NUM_WORKERS": NUM_WORKERS,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "SCHEDULER": SCHEDULER,
    "SCHEDULER_STEP": SCHEDULER_STEP,
    "SCHEDULER_REDUCTION": SCHEDULER_REDUCTION,
    "NUM_EPOCHS": NUM_EPOCHS,
    "OPTIMIZER": OPTIMIZER,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "IMAGE_SIZE": IMAGE_SIZE,
    "IMAGENET_MEAN": str(IMAGENET_MEAN),
    "IMAGENET_STD": str(IMAGENET_STD),
    "CROP_SCALE": str(CROP_SCALE),
    "ROTATION_DEGREES": ROTATION_DEGREES,
    "HORIZONTAL_FLIP_PROB": HORIZONTAL_FLIP_PROB,
    "BRIGHTNESS": BRIGHTNESS,
    "CONTRAST": CONTRAST,
    "SATURATION": SATURATION,
    "HUE": HUE,
    "CNN_CONV_LAYERS": str(CNN_CONV_LAYERS),
    "CNN_DROPOUT_RATE": CNN_DROPOUT_RATE,
    "CNN_USE_BATCHNORM": CNN_USE_BATCHNORM,
    "CNN_POOLING_TYPE": CNN_POOLING_TYPE,
    "CNN_DENSE_NEURONS": CNN_DENSE_NEURONS,
}

writer.add_hparams(hparams, {}, run_name=f"../{EXPERIMENT_NAME}")

# TRANSFORMS
train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=CROP_SCALE),
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        transforms.RandomRotation(ROTATION_DEGREES),
        transforms.ColorJitter(
            brightness=BRIGHTNESS, contrast=CONTRAST, saturation=SATURATION, hue=HUE
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# DATALOADER
train_loader = get_dataloader(
    CSV_PATH, "train", BATCH_SIZE, NUM_WORKERS, train_transforms
)
val_loader = get_dataloader(CSV_PATH, "val", BATCH_SIZE, NUM_WORKERS, val_transforms)

# MODEL
model = CustomCNN(
    input_channels=3,
    num_classes=NUM_CLASSES,
    input_size=(IMAGE_SIZE, IMAGE_SIZE),
    conv_layers=CNN_CONV_LAYERS,  # 3 capas convolucionales
    dropout_rate=CNN_DROPOUT_RATE,
    use_batchnorm=CNN_USE_BATCHNORM,
    pooling_type=CNN_POOLING_TYPE,
    dense_neurons=CNN_DENSE_NEURONS,
).to(device)


# OPTIMIZER AND LOSS FUNCTION
criterion = nn.CrossEntropyLoss()
optimizer = select_optimizer(model, OPTIMIZER, LEARNING_RATE, WEIGHT_DECAY)
if SCHEDULER == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_REDUCTION
    )
elif SCHEDULER == "plateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_REDUCTION,
        patience=SCHEDULER_STEP,
        verbose=True,
    )

# CHECKPOINT LOAD
if FROM_CHECKPOINT:
    assert (
        CHECKPOINT_PATH is not None
    ), "You must provide the path to the checkpoint if FROM_CHECKPOINT is True"
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    starting_epoch = checkpoint["epoch"]
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
else:
    starting_epoch = 0

# TRAINING LOOP
for epoch in range(starting_epoch, NUM_EPOCHS + starting_epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
        current_lr = optimizer.param_groups[0][
            "lr"
        ]  # Asumiendo que usas un solo grupo de parámetros
        # Imprimir progreso cada N batches
        if (batch_idx + 1) % 10 == 0:  # Muestra la métrica cada 10 batches

            # Sobrescribir la línea en la terminal
            sys.stdout.write(
                f"\rEpoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                f"Train Loss: {running_loss / (batch_idx+1):.4f}, "
                f"Train Acc: {batch_accuracy:.4f}, "
                f"Learning Rate: {current_lr:.6f}"  # Agrega el learning rate
            )
            sys.stdout.flush()

    # Calcular métricas en entrenamiento y validación
    train_metrics = compute_metrics(model, train_loader, criterion, device)
    val_metrics = compute_metrics(model, val_loader, criterion, device)
    if SCHEDULER == "plateau":
        scheduler.step(val_metrics["loss"])
    else:
        scheduler.step()
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
            "scheduler_state_dict": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        checkpoint_path,
    )

# Cerrar TensorBoard
writer.close()
