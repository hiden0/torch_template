import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import sys
import os
import torch.nn as nn
import shutil
import yaml
import argparse
import optuna

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_dir, ".."))

from models.custom_CNN import CustomCNN
from data.PetImages_dataset import get_dataloader
from engine.metrics import compute_metrics, plot_confusion_matrix
from engine.training_utils import (
    select_optimizer,
    load_config,
    setup_device,
    setup_directories,
    save_hparams,
    save_checkpoint,
)


def train_model(config, trial=None):
    device = setup_device()
    log_dir, checkpoint_dir = setup_directories(config, base_dir, trial)

    writer = SummaryWriter(log_dir=log_dir)
    save_hparams(writer, config, log_dir)

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config["IMAGE_SIZE"], scale=config["CROP_SCALE"]
            ),
            transforms.RandomHorizontalFlip(p=config["HORIZONTAL_FLIP_PROB"]),
            transforms.RandomRotation(config["ROTATION_DEGREES"]),
            transforms.ColorJitter(
                brightness=config["BRIGHTNESS"],
                contrast=config["CONTRAST"],
                saturation=config["SATURATION"],
                hue=config["HUE"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=eval(str(config["IMAGENET_MEAN"])),
                std=eval(str(config["IMAGENET_STD"])),
            ),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((config["IMAGE_SIZE"], config["IMAGE_SIZE"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=eval(str(config["IMAGENET_MEAN"])),
                std=eval(str(config["IMAGENET_STD"])),
            ),
        ]
    )

    train_loader = get_dataloader(
        config["CSV_PATH"],
        "train",
        config["BATCH_SIZE"],
        config["NUM_WORKERS"],
        train_transforms,
    )
    val_loader = get_dataloader(
        config["CSV_PATH"],
        "val",
        config["BATCH_SIZE"],
        config["NUM_WORKERS"],
        val_transforms,
    )

    model = CustomCNN(
        input_channels=3,
        num_classes=config["NUM_CLASSES"],
        input_size=(config["IMAGE_SIZE"], config["IMAGE_SIZE"]),
        conv_layers=config["CNN_CONV_LAYERS"],
        dropout_rate=config["CNN_DROPOUT_RATE"],
        use_batchnorm=config["CNN_USE_BATCHNORM"],
        pooling_type=config["CNN_POOLING_TYPE"],
        dense_neurons=config["CNN_DENSE_NEURONS"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = select_optimizer(
        model, config["OPTIMIZER"], config["LEARNING_RATE"], config["WEIGHT_DECAY"]
    )
    if config["SCHEDULER"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["SCHEDULER_STEP"],
            gamma=config["SCHEDULER_REDUCTION"],
        )
    elif config["SCHEDULER"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["SCHEDULER_REDUCTION"],
            patience=config["SCHEDULER_STEP"],
            verbose=True,
        )

    if config["FROM_CHECKPOINT"]:
        assert (
            config["CHECKPOINT_PATH"] is not None
        ), "You must provide the path to the checkpoint if FROM_CHECKPOINT is True"
        checkpoint = torch.load(
            config["CHECKPOINT_PATH"], weights_only=False, map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"]
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        starting_epoch = 0

    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(starting_epoch, config["NUM_EPOCHS"] + starting_epoch):
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
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            batch_accuracy = correct / total if total > 0 else 0.0
            current_lr = optimizer.param_groups[0]["lr"]
            if (batch_idx + 1) % 10 == 0:
                sys.stdout.write(
                    f"\rEpoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] Train Loss: {running_loss / (batch_idx+1):.4f}, Train Acc: {batch_accuracy:.4f}, Learning Rate: {current_lr:.6f}"
                )
                sys.stdout.flush()

        train_metrics = compute_metrics(model, train_loader, criterion, device)
        val_metrics = compute_metrics(model, val_loader, criterion, device)
        if config["SCHEDULER"] == "plateau":
            scheduler.step(val_metrics["loss"])
        else:
            scheduler.step()

        train_cm_image = plot_confusion_matrix(
            train_metrics["confusion_matrix"],
            class_names=[str(i) for i in range(config["NUM_CLASSES"])],
        )
        val_cm_image = plot_confusion_matrix(
            val_metrics["confusion_matrix"],
            class_names=[str(i) for i in range(config["NUM_CLASSES"])],
        )

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
        writer.add_image(
            "Confusion_Matrix/Train", train_cm_image, epoch, dataformats="HWC"
        )
        writer.add_image(
            "Confusion_Matrix/Validation", val_cm_image, epoch, dataformats="HWC"
        )

        print(
            f"\nEpoch {epoch+1}: Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Train Precision: {train_metrics['precision']:.4f}, Train Recall: {train_metrics['recall']:.4f}, Train F1: {train_metrics['f1_score']:.4f}"
        )
        print(
            f"          Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1_score']:.4f}"
        )

        save_checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            train_metrics,
            val_metrics,
            checkpoint_dir,
        )

        # Early Stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.get("EARLY_STOPPING_PATIENCE", 5):
            print(f"Early stopping at epoch {epoch+1}")
            break

        if trial:
            trial.report(val_metrics["accuracy"], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    writer.close()
    return val_metrics["accuracy"]
