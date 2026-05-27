# Calibración de Cámara y Detección ArUco — DJI Tello

Proyecto final para calibrar la cámara del dron **DJI Tello** y detectar marcadores **ArUco** usando OpenCV.

## Descripción

El flujo completo cubre tres etapas:

1. **Captura de frames** desde el dron para construir el conjunto de calibración.
2. **Calibración intrínseca** de la cámara con un tablero de ajedrez.
3. **Detección de marcadores ArUco** usando los parámetros obtenidos.

También incluye utilidades para generar los patrones de calibración y verificar el resultado visualmente.

---

## Estructura del proyecto

```
FinalProyect/
├── captureframes.py        # Captura frames desde el Tello (guarda en calib_frames/)
├── calibracion.py          # Calibra la cámara con los frames capturados
├── arcucodetection.py      # Genera un marcador ArUco (DICT_4X4_50, ID=0)
├── checkerboard.py         # Genera la imagen del tablero de ajedrez (10×7 cuadros)
│
├── checkerboard_9x6.png    # Tablero generado (listo para imprimir)
├── aruco_id0.png           # Marcador ArUco ID=0 generado (300×300 px)
│
├── camera_params.npz       # Parámetros de calibración (cámara principal)
├── camera_paramsSoni.npz   # Parámetros de calibración (cámara Soni)
│
├── calib_frames/           # Frames capturados para calibración (43 imágenes)
└── calib_framesSoni/       # Frames alternativos — cámara Soni (34 imágenes)
```

---

## Requisitos

```
Python >= 3.8
opencv-contrib-python
djitellopy
numpy
```

Instalar dependencias:

```bash
pip install opencv-contrib-python djitellopy numpy
```

> **Nota:** se necesita `opencv-contrib-python` (no `opencv-python`) para tener soporte de ArUco.

---

## Flujo de uso

### 0. Generar los patrones de calibración (solo una vez)

```bash
# Genera checkerboard_9x6.png (10×7 cuadros, 80 px/cuadro)
python checkerboard.py

# Genera aruco_id0.png (marcador ArUco DICT_4X4_50, ID=0)
python arcucodetection.py
```

Imprime `checkerboard_9x6.png` y mide el lado real de cada cuadro en metros — ese valor va en `SQUARE_SIZE` dentro de `calibracion.py`.

### 1. Capturar frames de calibración

Conecta el Tello por Wi-Fi y ejecuta:

```bash
python captureframes.py
```

| Tecla | Acción |
|-------|--------|
| `s`   | Guardar frame actual en `calib_frames/` |
| `q`   | Salir |

Captura **al menos 20–30 frames** variando la inclinación, rotación y distancia del tablero respecto a la cámara.

### 2. Calibrar la cámara

```bash
python calibracion.py
```

El script procesa todas las imágenes en `calib_frames/`, muestra las detecciones y guarda el resultado en `camera_params.npz` (matrices `K` y `dist`).

- RMS error < 1.0 px → calibración válida.
- Si el error es mayor, captura más frames con mayor variedad de poses.

### 3. Verificar la calibración (script histórico)

El archivo `.history/verifycalib_20260513121852.py` muestra en tiempo real la imagen original vs. la imagen corregida (undistorted) desde el Tello:

```bash
python .history/verifycalib_20260513121852.py
```

### (Opcional) Visor de cámara dual

`.history/tello_dual_camera_20260513113826.py` permite alternar entre la cámara frontal y la cámara inferior del Tello:

```bash
python .history/tello_dual_camera_20260513113826.py
```

| Tecla | Acción |
|-------|--------|
| `f`   | Cámara frontal |
| `d`   | Cámara inferior (B&N) |
| `q`   | Salir |

---

## Parámetros de calibración

Los archivos `.npz` contienen:

| Variable | Descripción |
|----------|-------------|
| `K`      | Matriz intrínseca de la cámara (3×3) |
| `dist`   | Coeficientes de distorsión (k1, k2, p1, p2, k3) |

Carga en tu código con:

```python
import numpy as np
data = np.load("camera_params.npz")
K, dist = data["K"], data["dist"]
```

---

## Configuración del tablero

| Parámetro | Valor por defecto |
|-----------|-------------------|
| Esquinas internas | 9 × 6 |
| Tamaño de cuadro | 0.025 m (ajustar según impresión) |
| Patrón generado | 10 × 7 cuadros, 80 px/cuadro |
