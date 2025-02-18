# 🔥 Optimización en PyTorch: Explicación de los principales optimizadores  

En este documento, se explican los diferentes optimizadores seleccionables en el **torch_template**, junto con sus parámetros clave y cómo ajustarlos (con valores recomendados) para mejorar el entrenamiento de modelos de **Deep Learning**.

---

## 🟢 1. Adam (Adaptive Moment Estimation)  

El optimizador **Adam** combina las ventajas de **SGD con momentum** y **RMSprop**, adaptando la tasa de aprendizaje de cada parámetro de manera independiente.

```python
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=0.001,
    weight_decay=0, 
)
```

### 🔹 **Parámetros de Adam**  
- **`lr` (Learning Rate, por defecto `0.001`)**  
  - Controla la magnitud de los pasos en la dirección del gradiente.  
  - Valores recomendados: `0.0001 - 0.001`  
- **`weight_decay` (default=`0`)**  
  - Añade regularización L2, pero **AdamW es mejor alternativa**.   

---

## 🔵 2. AdamW (Adam con mejor regularización L2)  

Corrige el problema de **weight decay** en Adam, evitando que se aplique incorrectamente.

```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0.001,
    weight_decay=0
)
```

✅ **Recomendado si quieres evitar sobreajuste con regularización L2.**

---

## 🟠 3. SGD (Stochastic Gradient Descent) con Momentum  

El **descenso de gradiente estocástico (SGD)** es más simple y generalmente ofrece mejor **generalización** que Adam.

```python
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.01, 
    momentum=0.9, 
    weight_decay=0.0001, 
)
```

### 🔹 **Parámetros de SGD**  
- **`lr`**: Define el tamaño de los pasos en la actualización de pesos.  
- **`momentum` (default=`0.9`)**:  
  - Reduce la variabilidad en la dirección del gradiente. (Configurable solo desde linea de código)  
  - Valores recomendados: `0.9`  
- **`weight_decay`**: Regularización L2 para evitar sobreajuste.   

---

## 🔴 4. RMSprop (Root Mean Square Propagation)  

RMSprop es útil para problemas donde los gradientes oscilan mucho, como **redes recurrentes**.

```python
optimizer = torch.optim.RMSprop(
    model.parameters(), 
    lr=0.001, 
    alpha=0.99, 
    eps=1e-8, 
    weight_decay=0, 
    momentum=0.9
)
```

### 🔹 **Parámetros de RMSprop**  
- **`lr`**: Tasa de aprendizaje.  
- **`alpha` (default=`0.99`)**:  
  - Define cuánto contribuyen los gradientes antiguos en el nuevo ajuste. (Configurable solo desde linea de código)  
  - Valores recomendados: `0.9 - 0.99`  
- **`eps`**: Término de estabilidad numérica. (Configurable solo desde linea de código)  
- **`weight_decay`**: Regularización L2.  
- **`momentum`**: Permite que el optimizador almacene información de los gradientes pasados. (Configurable solo desde linea de código)   

✅ **RMSprop suele funcionar mejor que Adam en problemas con gradientes muy oscilantes.**

---

## 🟣 5. Adagrad (Adaptive Gradient Algorithm)  

Adagrad ajusta la tasa de aprendizaje para cada parámetro de forma individual, útil en **datasets con características raras**.

```python
optimizer = torch.optim.Adagrad(
    model.parameters(), 
    lr=0.01, 
    lr_decay=0, 
    weight_decay=0
)
```

### 🔹 **Parámetros de Adagrad**  
- **`lr`**: Define el tamaño inicial del paso de actualización.  
- **`lr_decay` (default=`0`)**:  
  - Controla cómo decae el learning rate con el tiempo. (Configurable solo desde linea de código) 
- **`weight_decay`**: Regularización L2.  

🚨 **Problema:** Reduce demasiado el `lr` con el tiempo, lo que puede hacer que el entrenamiento se detenga prematuramente.

---

## 🟤 6. Adadelta (Mejora de Adagrad)  

Elimina la necesidad de definir un **learning rate**, adaptándolo automáticamente.

```python
optimizer = torch.optim.Adadelta(
    model.parameters(), 
    rho=0.9, 
    eps=1e-6
)
```

### 🔹 **Parámetros de Adadelta**  
- **`rho` (default=`0.9`)**:  
  - Controla la memoria del gradiente pasado. (Configurable solo desde linea de código) 
- **`eps`**: Término de estabilidad numérica. (Configurable solo desde linea de código) 

✅ **Útil si no quieres preocuparte por el ajuste del `lr`.**

---



---

# 📊 Comparación rápida  

| Optimizador | Velocidad | Generalización | Estabilidad | Cuándo usarlo |
|-------------|-----------|----------------|-------------|---------------|
| **Adam** | 🔹🔹🔹 | 🔹🔹 | 🔹🔹🔹 | Casos generales |
| **AdamW** | 🔹🔹🔹 | 🔹🔹🔹 | 🔹🔹🔹 | Evitar sobreajuste |
| **SGD+Momentum** | 🔹🔹 | 🔹🔹🔹🔹 | 🔹🔹 | Modelos grandes |
| **RMSprop** | 🔹🔹🔹 | 🔹🔹 | 🔹🔹🔹🔹 | RNNs, gradientes oscilantes |
| **Adagrad** | 🔹 | 🔹🔹🔹 | 🔹🔹🔹 | NLP, datos dispersos |
| **Adadelta** | 🔹🔹🔹 | 🔹🔹🔹 | 🔹🔹🔹 | Evitar ajuste del LR |

---

# 🚀 Resumen 

- Si **Adam te funciona bien**, prueba **AdamW** para mejor regularización.  
- Si **el modelo sobreajusta**, usa **SGD con momentum**.  
- Si **los gradientes oscilan mucho**, usa **RMSprop**.   

📌 **¿Cuál de estos quieres probar en tu caso?** 😃