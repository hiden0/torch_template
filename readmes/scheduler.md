# 🔹 Guía Completa sobre Schedulers en PyTorch

Los schedulers en PyTorch permiten ajustar la tasa de aprendizaje (learning rate, LR) durante el entrenamiento para mejorar la convergencia y evitar que el modelo se quede atrapado en mínimos locales.

## 🎯 **¿Por qué usar un Scheduler?**
- Evita que el LR sea demasiado alto y cause oscilaciones en la pérdida.
- Previene que el LR sea demasiado bajo y ralentice la convergencia.
- Mejora la generalización del modelo.
- Permite entrenar más eficientemente reduciendo el LR en el momento adecuado.

## 🔥 **Principales Schedulers en PyTorch**

### 1️⃣ **StepLR**
Reduce el LR en pasos fijos de épocas.
```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR cada 10 epochs

for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```
**🔹 Parámetros:**
- `step_size`: Número de épocas tras las cuales se reduce el LR.
- `gamma`: Factor de reducción (ej. `0.1` reduce el LR a un 10% de su valor actual).

---
### 2️⃣ **ReduceLROnPlateau**
Reduce el LR cuando la métrica de validación deja de mejorar.
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)
```
**🔹 Parámetros:**
- `mode`: `'min'` para minimizar la pérdida, `'max'` para métricas como accuracy.
- `factor`: Multiplica el LR por este valor cuando se reduce.
- `patience`: Número de épocas sin mejora antes de reducir el LR.
- `verbose`: Muestra logs cuando el LR cambia.

---
### 3️⃣ **ExponentialLR**
Reduce el LR exponencialmente en cada epoch.
```python
from torch.optim.lr_scheduler import ExponentialLR

scheduler = ExponentialLR(optimizer, gamma=0.95)  # Reduce LR un 5% en cada epoch

for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```
**🔹 Parámetros:**
- `gamma`: Factor por el cual se multiplica el LR en cada epoch.

---
## 📌 **¿Cuál elegir?**
| Tipo de Scheduler  | Cuándo Usarlo |
|--------------------|------------------------------------------------|
| `StepLR` | Si quieres reducir el LR en épocas fijas. |
| `ReduceLROnPlateau` | Si quieres reducir el LR cuando la pérdida deje de mejorar. |
| `ExponentialLR` | Si necesitas un decrecimiento constante en cada epoch. |

