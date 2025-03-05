import sys
import os
import argparse

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_dir, ".."))

from engine.training_utils import load_config
from engine.train import train_model


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
    train_model(load_config(args.config))
