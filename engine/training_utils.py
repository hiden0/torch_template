import torch
import torch.optim as optim
import torch.nn as nn
import os
import shutil
import yaml
from torch.utils.tensorboard import SummaryWriter
import sys


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    for i in range(len(config["CNN_CONV_LAYERS"])):
        config["CNN_CONV_LAYERS"][i] = tuple(config["CNN_CONV_LAYERS"][i])
    return config


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


def setup_directories(config, base_dir, trial):
    log_dir = os.path.join(
        base_dir, f"../runs/training_logs/{config['EXPERIMENT_NAME']}"
    )
    checkpoint_dir = os.path.join(
        base_dir, f"../runs/checkpoints/{config['EXPERIMENT_NAME']}"
    )

    if (
        os.path.exists(checkpoint_dir)
        and not config["FROM_CHECKPOINT"]
        and (trial is None or trial._trial_id == 0)
    ):
        response = (
            input(
                f"Experiment '{config['EXPERIMENT_NAME']}' already exists. Do you want to overwrite it? (y/n): "
            )
            .strip()
            .lower()
        )
        if response != "y":
            print("Exiting script without overwriting experiment.")
            sys.exit()
        else:
            print(
                f"The old experiment '{config['EXPERIMENT_NAME']}' will be overwritten."
            )
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            shutil.rmtree(log_dir, ignore_errors=True)

    # Modificación para incluir el número del trial en el nombre del directorio de registro
    if trial:
        log_dir = os.path.join(log_dir, f"trial_{trial.number}")
        checkpoint_dir = os.path.join(checkpoint_dir, f"trial_{trial.number}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir, checkpoint_dir


def save_hparams(writer, config, log_dir):
    hparams = {key: str(value) for key, value in config.items()}
    writer.add_hparams(hparams, {}, run_name=f"{log_dir}/../{log_dir.split('/')[-1]}")


def save_checkpoint(
    epoch, model, optimizer, scheduler, train_metrics, val_metrics, checkpoint_dir
):
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
