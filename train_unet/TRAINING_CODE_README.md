# Código de Entrenamiento U-Net - Guía de Uso

## 📋 Descripción

Este directorio contiene **8 archivos de código Python** organizados por celdas del notebook `train_unet_notebook.ipynb`. Cada archivo corresponde a una sección específica del pipeline de entrenamiento.

## 📁 Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `cell_01_imports.py` | Imports de bibliotecas necesarias |
| `cell_02_hyperparameters.py` | Configuración de hiperparámetros |
| `cell_03_dataset.py` | Clase SegmentationDataset y DataLoaders |
| `cell_04_model.py` | Arquitectura OptimizedUNet |
| `cell_05_training_functions.py` | Funciones de entrenamiento y métricas |
| `cell_06_training_loop.py` | Bucle principal de entrenamiento |
| `cell_07_visualization.py` | Visualización de predicciones |
| `cell_08_onnx_export.py` | Exportación a formato ONNX |

## 🚀 Cómo Usar

### Opción 1: Copiar/Pegar en el Notebook

1. **Abre** `train_unet_notebook.ipynb` en Jupyter
2. **Crea una nueva celda** después de las celdas existentes
3. **Copia el contenido** de `cell_01_imports.py`
4. **Pega** en la nueva celda
5. **Repite** los pasos 2-4 para los archivos `cell_02` a `cell_08`
6. **Ejecuta** las celdas en orden

### Opción 2: Ejecutar como Script Python

Si prefieres ejecutar todo de una vez:

```bash
cd /home/hackbrian/Documents/ACC_Development/Development/ros2/src/qcar2_LaneSeg-ACC/train_unet

# Concatenar todos los archivos
cat cell_01_imports.py \
    cell_02_hyperparameters.py \
    cell_03_dataset.py \
    cell_04_model.py \
    cell_05_training_functions.py \
    cell_06_training_loop.py \
    cell_07_visualization.py \
    cell_08_onnx_export.py > unet_training_complete.py

# Ejecutar
python unet_training_complete.py
```

## 📊 Salidas Esperadas

Después de ejecutar el código completo, se generarán los siguientes archivos:

| Archivo | Descripción |
|---------|-------------|
| `best_model.pth` | Modelo PyTorch con menor pérdida de validación |
| `lane_unet.onnx` | Modelo exportado para Isaac ROS |
| `training_curves.png` | Gráficas de pérdida, accuracy y mIoU |
| `predictions_visualization.png` | Comparación visual de predicciones |

## ⚙️ Configuración Crítica para Isaac ROS

El código está configurado específicamente para ser compatible con Isaac ROS:

- **Resolución de entrada:** 640x480 (ancho x alto)
- **Opset ONNX:** versión 11
- **Nombres de tensores:**
  - Input: `input_tensor`
  - Output: `output_tensor`
- **Clases de salida:** 4 (fondo, camino, líneas, bordes)

## 🎨 Clases de Segmentación

| ID | Nombre | Color | Descripción |
|----|--------|-------|-------------|
| 0 | fondo, vereda, obstáculo | Negro | Background/Sidewalk/Obstacles |
| 1 | camino, asfalto, road | Azul | Drivable road/Asphalt |
| 2 | línea, lane | Amarillo | Traffic lane markings |
| 3 | borde, edge | Rojo | Road edges |

## 🔧 Pesos de Clase

Los pesos asignados en `CrossEntropyLoss` son:

```python
CLASS_WEIGHTS = [0.1, 1.0, 10.0, 1.0]
```

- **Clase 0 (Fondo):** 0.1 (muy frecuente)
- **Clase 1 (Camino):** 1.0 (frecuente)
- **Clase 2 (Líneas):** 10.0 ⚠️ **Peso alto para contrarrestar desbalance**
- **Clase 3 (Bordes):** 1.0 (moderadamente frecuente)

## 📈 Métricas Implementadas

- **Pixel Accuracy:** Porcentaje de píxeles correctamente clasificados
- **Mean IoU (mIoU):** Intersection over Union promedio de todas las clases
- **Loss:** CrossEntropyLoss con pesos de clase

## 🎯 Parámetros Ajustables

En `cell_02_hyperparameters.py` puedes modificar:

```python
BATCH_SIZE = 8           # Tamaño del batch (ajustar según GPU)
LEARNING_RATE = 1e-4     # Tasa de aprendizaje
NUM_EPOCHS = 20          # Número de épocas (20-50 recomendado)
```

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
**Solución:** Reduce `BATCH_SIZE` en `cell_02_hyperparameters.py`

### Error: "Dataset not found"
**Solución:** Verifica que existan las carpetas:
- `training_data/dataset_images/train/images`
- `training_data/dataset_images/train/masks`
- `training_data/dataset_images/val/images`
- `training_data/dataset_images/val/masks`

### Las predicciones son malas
**Soluciones:**
- Aumentar `NUM_EPOCHS` (probar con 50)
- Ajustar `CLASS_WEIGHTS` según tu dataset
- Verificar calidad del dataset

## 📚 Próximos Pasos

1. **Validar ONNX con TensorRT:**
   ```bash
   /usr/src/tensorrt/bin/trtexec \
     --onnx=lane_unet.onnx \
     --saveEngine=lane_unet.plan \
     --fp16 \
     --verbose
   ```

2. **Integrar con Isaac ROS:**
   - Copiar `lane_unet.onnx` a tu workspace de Isaac ROS
   - Configurar el nodo de inferencia

3. **Probar en QCar2:**
   - Desplegar el modelo en el robot
   - Validar rendimiento en tiempo real

## 📝 Notas Importantes

> [!WARNING]
> **No modifiques las dimensiones de imagen (640x480)**  
> Esto rompería la compatibilidad con Isaac ROS

> [!IMPORTANT]
> **Guarda `best_model.pth` antes de exportar a ONNX**  
> Este es tu punto de control para reanudar entrenamiento

> [!TIP]
> **Usa GPU para entrenamiento**  
> El código detectará automáticamente CUDA si está disponible

## 🛠️ Requisitos

```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm
pip install onnx  # Opcional, para verificación
```

## 👨‍💻 Autor

Código generado para el proyecto QCar2 Lane Segmentation - Conducción Autónoma Simulada
