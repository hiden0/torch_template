import os
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_dataset_csv(dataset_path, output_csv, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Genera un archivo CSV con las rutas de las imágenes, sus etiquetas y el conjunto al que pertenecen (train/val/test).

    Parámetros:
        dataset_path (str): Ruta del dataset.
        output_csv (str): Ruta donde se guardará el archivo CSV.
        labels (dict): Diccionario que mapea nombres de carpetas a etiquetas.
        train_ratio (float): Proporción del conjunto de entrenamiento.
        val_ratio (float): Proporción del conjunto de validación.
        test_ratio (float): Proporción del conjunto de prueba.
        random_state (int): Semilla para la división aleatoria.
    """
    # Verificar que las proporciones sumen 1
    if not (abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6):
        raise ValueError("Las proporciones de train, val y test deben sumar 1.")

    # Lista para almacenar los datos
    data = []

    # Recorrer las carpetas
    for label_name, label_value in labels.items():
        folder_path = os.path.join(dataset_path, label_name)
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                # Obtener la ruta completa de la imagen
                image_path = os.path.join(root, file_name)
                # Añadir la información a la lista
                data.append([image_path, label_value])

    # Crear un DataFrame con pandas
    df = pd.DataFrame(data, columns=['path', 'label'])

    # Dividir el dataset en train, val y test
    train_df, test_val_df = train_test_split(df, test_size=(1 - train_ratio), random_state=random_state)
    val_df, test_df = train_test_split(test_val_df, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_state)

    # Asignar el conjunto correspondiente
    train_df['set'] = 'train'
    val_df['set'] = 'val'
    test_df['set'] = 'test'

    # Combinar los DataFrames
    final_df = pd.concat([train_df, val_df, test_df])

    # Guardar el DataFrame en un archivo CSV
    final_df.to_csv(output_csv, index=False)

    print(f"CSV generado en {output_csv}")

if __name__ == "__main__":
    # Parámetros configurables
    dataset_path = "/srv/hdd2/javber/example_dataset/PetImages"
    output_csv = "/srv/hdd2/javber/dataset.csv"
    labels = {"Dog": 0, "Cat": 1}  # Puedes modificar esto según tus carpetas y etiquetas
    train_ratio = 0.7  # 70% para entrenamiento
    val_ratio = 0.15   # 15% para validación
    test_ratio = 0.15  # 15% para prueba
    random_state = 42  # Semilla para reproducibilidad

    # Llamar a la función principal
    generate_dataset_csv(dataset_path, output_csv, labels, train_ratio, val_ratio, test_ratio, random_state)