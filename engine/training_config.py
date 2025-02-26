import torch
import torch.optim as optim
import torch.nn as nn


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
