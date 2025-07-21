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
from engine.train import train_model


def objective(trial, config_path):
    config = load_config(config_path)

    ### TRAIN HPARAMS ###
    config["BATCH_SIZE"] = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    config["LEARNING_RATE"] = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    config["SCHEDULER"] = trial.suggest_categorical("scheduler", ["step", "plateau"])
    config["SCHEDULER_STEP"] = trial.suggest_int("scheduler_step", 1, 10)
    config["SCHEDULER_REDUCTION"] = trial.suggest_uniform(
        "scheduler_reduction", 0.1, 0.99
    )
    # config["NUM_EPOCHS"] = trial.suggest_int("num_epochs", 2, 5)
    config["OPTIMIZER"] = trial.suggest_categorical(
        "optimizer", ["adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta"]
    )

    config["WEIGHT_DECAY"] = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    config["IMAGE_SIZE"] = trial.suggest_categorical("image_size", [64, 128, 256])

    ### DATA AUGMENTATION ###
    config["CROP_SCALE"] = [trial.suggest_uniform("crop_scale_min", 0.5, 1.0), 1.0]
    config["ROTATION_DEGREES"] = trial.suggest_int("rotation_degrees", 0, 45)
    config["HORIZONTAL_FLIP_PROB"] = trial.suggest_uniform(
        "horizontal_flip_prob", 0.0, 1.0
    )
    config["BRIGHTNESS"] = trial.suggest_uniform("brightness", 0.0, 0.5)
    config["CONTRAST"] = trial.suggest_uniform("contrast", 0.0, 0.5)
    config["SATURATION"] = trial.suggest_uniform("saturation", 0.0, 0.5)
    config["HUE"] = trial.suggest_uniform("hue", 0.0, 0.5)

    ### PRETRAINED CNN CONFIG ###
    config["MODEL"] = trial.suggest_categorical(
        "model",
        [
            "custom_cnn",
            "resnet50",
            "efficientnetb0",
            "mobilenetv2",
            "vgg16",
            "densenet121",
            #"inceptionv3",
        ],
    )
    config["PRETRAINED"] = trial.suggest_categorical("pretrained", [True, False])
    config["NUM_TRAIN_LAYERS"] = trial.suggest_categorical(
        "num_train_layers", [1, 2, 3, "all"]
    )

    ###CUSTOM CNN CONFIG (ONLY AVAILABLE WITH CUSTOM_CNN) ###
    config["CNN_DROPOUT_RATE"] = trial.suggest_uniform("dropout_rate", 0.1, 0.5)
    config["CNN_USE_BATCHNORM"] = trial.suggest_categorical(
        "use_batchnorm", [True, False]
    )
    config["CNN_POOLING_TYPE"] = trial.suggest_categorical(
        "pooling_type", ["max", "avg"]
    )
    config["CNN_DENSE_NEURONS"] = trial.suggest_categorical(
        "dense_neurons", [64, 128, 256, 512]
    )

    # Número de capas y filtros en cada capa
    num_layers = trial.suggest_int("num_layers", 1, 4)

    # Definir el número máximo de filtros (potencia de 2)
    initial_layer_filters = trial.suggest_categorical(
        "conv_layer_0_filters", [16, 32, 64, 128, 256]
    )

    # Generar las capas respetando la regla de ir decreciendo y siendo múltiplo de 2
    config["CNN_CONV_LAYERS"] = []

    for i in range(num_layers):
        if i == 0:
            filters = initial_layer_filters
        else:
            filters = filters * 2
        config["CNN_CONV_LAYERS"].append([filters, 3])
    
    
    def flow_style_list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(list, flow_style_list_representer)
    
    config_path = f"/app/configs/{config['EXPERIMENT_NAME']}/trial_{trial.number}.yml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        total_items = len(config)
        # Iteramos sobre cada par (clave, valor) del diccionario
        for i, (key, value) in enumerate(config.items()):
            # Guardamos cada variable como un diccionario de un solo elemento
            yaml.dump({key: value}, f)
            
            # Añadimos un salto de línea después de cada variable, excepto la última
            # if i < total_items - 1:
            #     f.write('\n')
        
    resultado = train_model(config, trial)
    torch.cuda.empty_cache()

    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script para entrenar un modelo de clasificación multiclase."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Ruta al archivo de configuración YAML",
    )

    args = parser.parse_args()
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args.config), n_trials=30, n_jobs=1)
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_trial.params}")
    
        # Guardar la mejor configuración en un YAML
    best_config_path = "best_optuna_config.yaml"
    with open(best_config_path, "w") as f:
        yaml.dump(study.best_trial.params, f)
    print(f"Mejor configuración guardada en {best_config_path}")
