# 🏆 **Funciones de Pérdida para Clasificación en CNNs**  

Cuando entrenamos una **red convolucional** para tareas de clasificación, elegir la función de pérdida correcta es clave para mejorar la precisión del modelo.  

A continuación, se presentan las mejores funciones de pérdida para clasificación, dependiendo de si el problema es **binario, multiclase o desbalanceado**.

---

## 🟢 **1. CrossEntropyLoss (Clasificación Multiclase)**  

```python
criterion = nn.CrossEntropyLoss(weight=None, reduction='mean')
```

✅ **¿Cuándo usarla?**  
✔ Cuando hay **más de dos clases** (Ejemplo: clasificación de imágenes con 10 categorías).  
✔ Es la opción más común en tareas de clasificación general.  

🔍 **Detalles:**  
- No necesita `softmax`, ya que PyTorch lo aplica internamente.  
- Si las clases están desbalanceadas, se puede usar `weight` para asignar pesos a cada clase.  

---

## 🔵 **2. BCEWithLogitsLoss (Clasificación Binaria)**  

```python
criterion = nn.BCEWithLogitsLoss(pos_weight=None, reduction='mean')
```

✅ **¿Cuándo usarla?**  
✔ Cuando el problema es **binario** (Ejemplo: detectar si hay un objeto en la imagen o no).  
✔ Se usa cuando la **última capa del modelo tiene una única neurona**.  

🔍 **Detalles:**  
- No necesita `sigmoid`, ya que PyTorch lo aplica internamente.  
- Se puede usar `pos_weight` para ajustar datasets desbalanceados.  

---

## 🟠 **3. Focal Loss (Para Datos Desbalanceados)**  

📌 **Nota:** PyTorch no la tiene implementada por defecto, pero la hemos definido como una clase personalizada.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()
```

✅ **¿Cuándo usarla?**  
✔ Cuando el dataset está **muy desbalanceado** (Ejemplo: clasificación de imágenes con muy pocos ejemplos de una clase).  
✔ Reduce la influencia de las clases mayoritarias y enfatiza las clases minoritarias.  

🔍 **Detalles:**  
- `gamma`: Controla cuánto se reduce la penalización para clases mayoritarias.  
- `alpha`: Peso para la clase minoritaria.  

📌 **Ejemplo en CNNs:**  

```python
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

---

## 📊 **Comparación de las Mejores Funciones de Pérdida en Clasificación**  

| Función de Pérdida | Tipo de Clasificación | ¿Cuándo Usarla? |
|--------------------|----------------------|----------------|
| **CrossEntropyLoss** | Multiclase | Cuando hay más de dos clases |
| **BCEWithLogitsLoss** | Binaria | Cuando solo hay dos clases |
| **Focal Loss** | Binaria o Multiclase | Cuando hay un gran desbalance de clases |

---

## 🚀 **¿Cuál elegir para tu CNN?**  

✔ **Si tu CNN clasifica en más de dos clases →** `CrossEntropyLoss`  
✔ **Si tu CNN clasifica en dos clases →** `BCEWithLogitsLoss`  
✔ **Si tu dataset está muy desbalanceado →** `Focal Loss`  

---

