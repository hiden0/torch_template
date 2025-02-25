import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        input_size=(32, 32),  # Tamaño de la imagen de entrada (altura, ancho)
        conv_layers=[(32, 3), (64, 3)],  # [(num_filtros, kernel_size), ...]
        dropout_rate=0.5,
        use_batchnorm=True,
        pooling_type="max",  # "max" o "avg"
        dense_neurons=128,
    ):
        super(CustomCNN, self).__init__()

        self.layers = nn.ModuleList()
        in_channels = input_channels
        h, w = input_size  # Extraemos altura y ancho

        # Agregar capas convolucionales
        for out_channels, kernel_size in conv_layers:
            self.layers.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )
            if use_batchnorm:
                self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU())

            if pooling_type == "max":
                self.layers.append(nn.MaxPool2d(2))
                h, w = input_size  # Extraemos altura y ancho
            elif pooling_type == "avg":
                self.layers.append(nn.AvgPool2d(2))
                h, w = h // 2, w // 2  # Igual que MaxPool2d

            in_channels = out_channels  # Actualizar canales de entrada

        # Capa de aplanamiento
        self.flatten = nn.Flatten()

        # Capa totalmente conectada
        self.fc = nn.Linear(in_channels * h * w, dense_neurons)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(dense_neurons, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
