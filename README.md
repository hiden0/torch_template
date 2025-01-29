# torch_template

classification_project/
│── configs/               # Configuraciones y parámetros de entrenamiento
│   ├── default.yaml
│── data/                  # Archivos CSV con rutas de imágenes y etiquetas
│   ├── train.csv          # Rutas de imágenes y etiquetas de entrenamiento
│   ├── val.csv            # Rutas de imágenes y etiquetas de validación
│   ├── test.csv           # Rutas de imágenes y etiquetas de prueba (opcional)
│── models/                # Definición de modelos
│   ├── __init__.py
│   ├── resnet.py          # Ejemplo de modelo basado en ResNet
│   ├── custom_model.py    # Tu propio modelo personalizado
│── scripts/               # Scripts para entrenamiento y evaluación
│   ├── train.py           # Script principal de entrenamiento
│   ├── test.py            # Evaluación del modelo en el set de validación/test
│   ├── predict.py         # Predicción en imágenes nuevas
│── utils/                 # Funciones auxiliares
│   ├── dataset.py         # Definición de Dataset y DataLoader a partir del CSV
│   ├── transforms.py      # Transformaciones y preprocesamiento
│   ├── metrics.py         # Funciones para calcular métricas (accuracy, F1, etc.)
│── checkpoints/           # Pesos de modelos guardados
│── notebooks/             # Notebooks para experimentación
│── requirements.txt       # Lista de dependencias (PyTorch, torchvision, etc.)
│── README.md              # Descripción del proyecto
│── .gitignore             # Archivos a ignorar en Git (pesos, datasets, etc.)
