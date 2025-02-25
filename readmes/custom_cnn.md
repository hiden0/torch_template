

---

# 🧠 CustomCNN: Red Convolucional Configurable en PyTorch  

## 📌 **Descripción General**  
La clase `CustomCNN` permite crear redes neuronales convolucionales **altamente configurables** mediante parámetros de entrada. De esta forma, se pueden definir **capas convolucionales con diferentes tamaños de kernel, número de filtros, tipos de pooling, uso de BatchNorm, dropout y configuración de la capa densa**.  

Esta flexibilidad la hace ideal para **experimentación en clasificación de imágenes** sin necesidad de escribir una nueva arquitectura manualmente cada vez.  

---

## ⚙ **Componentes de la Red**  

1. **Capas Convolucionales**  
   - Se pueden definir **varias capas** con diferentes números de filtros y tamaños de kernel.  
   - Se puede aplicar **BatchNorm** después de cada convolución para estabilizar el entrenamiento.  
   - Se usa **ReLU** como activación no lineal.  

2. **Pooling**  
   - Se puede elegir entre **MaxPooling** y **AveragePooling** después de cada convolución.  
   - Reduce el tamaño de la imagen y ayuda a mejorar la eficiencia computacional.  

3. **Capa de Aplanamiento (`Flatten`)**  
   - Convierte la salida de las capas convolucionales en un **vector unidimensional** para la capa densa.  

4. **Capa Totalmente Conectada (`fc`)**  
   - Contiene **neuronas ajustables** que aprenden características globales de la imagen.  
   - Se usa **ReLU** como activación.  

5. **Dropout**  
   - Evita el sobreajuste eliminando aleatoriamente algunas neuronas en cada iteración.  

6. **Capa de Salida**  
   - Contiene **tantas neuronas como clases** en el problema de clasificación.  
   - No usa activación (en problemas de clasificación se aplicaría `Softmax` o `Sigmoid` después).  



---

## 🎛 **Parámetros Ajustables**  

| Parámetro        | Descripción |
|-----------------|-------------|
| `input_channels` | Número de canales de entrada (ej. 3 para RGB, 1 para escala de grises). |
| `num_classes` | Número de clases de salida. |
| `conv_layers` | Lista de tuplas `(num_filtros, tamaño_kernel)`. Define las capas convolucionales. |
| `dropout_rate` | Probabilidad de dropout en la capa densa. |
| `use_batchnorm` | Si `True`, añade BatchNorm después de cada capa convolucional. |
| `pooling_type` | Tipo de pooling: `"max"` (MaxPooling) o `"avg"` (AveragePooling). |
| `dense_neurons` | Número de neuronas en la capa completamente conectada. |

---

## 🔧 **Ejemplos de Modelos Configurados**  

### 📌 **Ejemplo 1: CNN Pequeña**
```python
model = CustomCNN(
    input_channels=3,
    num_classes=10,
    conv_layers=[(16, 3), (32, 3)],  # 2 capas convolucionales
    dropout_rate=0.3,
    use_batchnorm=True,
    pooling_type="max",
    dense_neurons=64,
)
```

#### 🖼 **Arquitectura Resultante**  
```
Entrada (3x32x32)  
→ Conv2D(16, 3x3) → BatchNorm → ReLU → MaxPool  
→ Conv2D(32, 3x3) → BatchNorm → ReLU → MaxPool  
→ Flatten  
→ Dense(64) → ReLU → Dropout(0.3)  
→ Dense(10) (Salida)
```

---

### 📌 **Ejemplo 2: CNN Más Profunda**
```python
model = CustomCNN(
    input_channels=1,
    num_classes=5,
    conv_layers=[(32, 3), (64, 3), (128, 3)],  # 3 capas convolucionales
    dropout_rate=0.5,
    use_batchnorm=True,
    pooling_type="avg",
    dense_neurons=256,
)
```

#### 🖼 **Arquitectura Resultante**  
```
Entrada (1x32x32)  
→ Conv2D(32, 3x3) → BatchNorm → ReLU → AvgPool  
→ Conv2D(64, 3x3) → BatchNorm → ReLU → AvgPool  
→ Conv2D(128, 3x3) → BatchNorm → ReLU → AvgPool  
→ Flatten  
→ Dense(256) → ReLU → Dropout(0.5)  
→ Dense(5) (Salida)
```

---

## 📊 **Visualización de Ejemplo**  
Aquí tienes un diagrama de una arquitectura generada con `CustomCNN` (3 capas convolucionales, max pooling, y BatchNorm):

```plaintext
        ┌──────────┐
Input → │ Conv2D(32, 3x3) │ → BatchNorm → ReLU → MaxPool
        ├──────────┤
        │ Conv2D(64, 3x3) │ → BatchNorm → ReLU → MaxPool
        ├──────────┤
        │ Conv2D(128, 3x3) │ → BatchNorm → ReLU → MaxPool
        ├──────────┤
        │ Flatten  │
        ├──────────┤
        │ Dense(256) → ReLU → Dropout(0.5) │
        ├──────────┤
        │ Output (num_classes) │
        └──────────┘
```
