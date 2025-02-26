# Data augmentation en el Script de Entrenamiento

El script de entrenamiento utiliza varias transformaciones de datos para mejorar la robustez del modelo y prevenir el sobreajuste. A continuación, se explican cada una de ellas, cómo configurarlas y cómo desactivarlas.

---

## 1. **RandomResizedCrop** (Recorte y redimensionado aleatorio)

**Definición en el script:**
```python
transforms.RandomResizedCrop(IMAGE_SIZE, scale=CROP_SCALE)
```

- **Propósito:** Recorta una región aleatoria de la imagen y la redimensiona a `IMAGE_SIZE`.
- **Configuración:**
  - `IMAGE_SIZE`: Dimensión final de la imagen tras el recorte.
  - `CROP_SCALE`: Tupla con el rango de escala del recorte aleatorio, ej. `(0.8, 1.0)`.
- **Desactivación:** Para desactivarlo, configurar `CROP_SCALE = (1.0, 1.0)`.

---

## 2. **RandomHorizontalFlip** (Volteo horizontal aleatorio)

**Definición en el script:**
```python
transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB)
```

- **Propósito:** Voltea la imagen horizontalmente con una probabilidad `p`.
- **Configuración:**
  - `HORIZONTAL_FLIP_PROB`: Probabilidad de aplicar el volteo, ej. `0.5` (50%).
- **Desactivación:** Configurar `HORIZONTAL_FLIP_PROB = 0.0`.

---

## 3. **RandomRotation** (Rotación aleatoria)

**Definición en el script:**
```python
transforms.RandomRotation(ROTATION_DEGREES)
```

- **Propósito:** Rota la imagen en un rango de grados aleatorio.
- **Configuración:**
  - `ROTATION_DEGREES`: Valor máximo de rotación en grados (positivo y negativo), ej. `15` grados.
- **Desactivación:** Configurar `ROTATION_DEGREES = 0`.

---

## 4. **ColorJitter** (Ajuste de brillo, contraste, saturación y tono)

**Definición en el script:**
```python
transforms.ColorJitter(
    brightness=BRIGHTNESS, contrast=CONTRAST, saturation=SATURATION, hue=HUE
)
```

- **Propósito:** Modifica aleatoriamente los valores de brillo, contraste, saturación y tono de la imagen.
- **Configuración:**
  - `BRIGHTNESS`: Factor de ajuste del brillo, ej. `0.2`.
  - `CONTRAST`: Factor de ajuste del contraste, ej. `0.2`.
  - `SATURATION`: Factor de ajuste de la saturación, ej. `0.2`.
  - `HUE`: Factor de ajuste del tono, ej. `0.1` (debe estar en el rango `[0, 0.5]`).
- **Desactivación:** Configurar `BRIGHTNESS = 0`, `CONTRAST = 0`, `SATURATION = 0` y `HUE = 0`.

---


## 5. **Normalize** (Normalización)

**Definición en el script:**
```python
transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
```

- **Propósito:** Normaliza los valores de los pixeles de la imagen usando la media y desviación estándar de ImageNet.
- **Configuración:**
  - `IMAGENET_MEAN`: Lista con la media para cada canal RGB, ej. `[0.485, 0.456, 0.406]`.
  - `IMAGENET_STD`: Lista con la desviación estándar para cada canal RGB, ej. `[0.229, 0.224, 0.225]`.
- **Desactivación:** Configurar `IMAGENET_MEAN = [0, 0, 0]` y `IMAGENET_STD = [1, 1, 1]`.

---

## 6. **Resize (Validación y Test)**

**Definición en el script:**
```python
transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
```

- **Propósito:** Ajusta la imagen al tamaño especificado para la validación y test.
- **Configuración:**
  - `IMAGE_SIZE`: Tamaño final de la imagen.
- **Desactivación:** No es recomendable desactivarlo, ya que garantiza que todas las imágenes tengan un tamaño uniforme.

---

