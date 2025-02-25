import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import warnings


class ImageDataset(Dataset):
    def __init__(self, csv_file, split="train", img_size=256):
        """
        Dataset para cargar imágenes desde un CSV.
        :param csv_file: Ruta al archivo CSV.
        :param split: 'train', 'val' o 'test' (según la columna 'split' del CSV).
        :param img_size: Tamaño al que se redimensionarán las imágenes.
        """
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["set"] == split].reset_index(drop=True)

        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            image = Image.open(img_path).convert("RGB")
            important_warnings = [
                warning for warning in w if issubclass(warning.category, UserWarning)
            ]
            if important_warnings:
                for warning in important_warnings:
                    print(f"Warning captured: {warning.message} at {img_path}")

        image = self.transform(image)

        return image, label


def get_dataloader(
    csv_path, split="train", batch_size=32, shuffle=True, num_workers=4, img_size=256
):
    """
    Crea un DataLoader basado en la partición de datos.
    :param csv_path: Ruta al archivo CSV.
    :param split: 'train', 'val' o 'test'.
    :param batch_size: Tamaño del batch.
    :param shuffle: Si los datos deben mezclarse.
    :param num_workers: Número de hilos de carga.
    :return: DataLoader listo para usar.
    """
    assert split in ["train", "val", "test"], "split debe ser 'train', 'val' o 'test'"

    dataset = ImageDataset(csv_file=csv_path, split=split, img_size=img_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if split == "train" else False),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
