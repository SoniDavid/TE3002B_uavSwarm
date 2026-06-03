# Calibración de Cámara, Detección ArUco y Control de Vuelo — DJI Tello

Proyecto final para calibrar la cámara del dron **DJI Tello**, detectar marcadores **ArUco** y controlar el vuelo de forma autónoma para seguir un marcador en tiempo real.

## Descripción

El proyecto cubre dos fases principales:

1. **Calibración y utilidades** — captura de frames, calibración intrínseca y generación de patrones.
2. **Sistema de control autónomo** — seguimiento en tiempo real de un marcador ArUco con visualización HUD y gráfica 3D.

---

## Estructura del proyecto

```
FinalProyect/
│
├── main/
│   └── files/
│       ├── main.py             # Punto de entrada — lanza TelloController
│       ├── tello_controller.py # Orquesta conexión, stream, hilos y display loop
│       ├── detector.py         # Detección ArUco + estimación de pose (solvePnP)
│       ├── controller.py       # Control RC proporcional (fb, yaw, up/down)
│       ├── hud.py              # Overlay OpenCV + gráfica 3D matplotlib
│       └── stream.py           # Buffer UDP — conserva solo el frame más reciente
│
├── captureframes.py            # Captura frames desde el Tello (guarda en calib_frames/)
├── checkerboard.py             # Genera la imagen del tablero de ajedrez (10×7 cuadros)
├── arucos/
│   ├── arucocreation.py        # Genera un marcador ArUco (DICT_4X4_50)
│   ├── calibracion.py          # Calibra la cámara con los frames capturados
│   └── detect_aruco.py         # Detección ArUco básica (standalone)
│
├── checkerboard_9x6.png        # Tablero generado (listo para imprimir)
├── arucos/aruco_id0.png        # Marcador ArUco ID=0 generado (300×300 px)
├── arucos/aruco_id1.png        # Marcador ArUco ID=1 generado (300×300 px)
│
├── camera_params.npz           # Parámetros de calibración (cámara principal)
├── camera_paramsSoni.npz       # Parámetros de calibración (cámara Soni)
│
├── calib_frames/               # Frames capturados para calibración (43 imágenes)
└── calib_framesSoni/           # Frames alternativos — cámara Soni (34 imágenes)
```

---

## Requisitos

```
Python >= 3.8
opencv-contrib-python
djitellopy
numpy
matplotlib
```

Instalar dependencias:

```bash
pip install opencv-contrib-python djitellopy numpy matplotlib
```

> **Nota:** se necesita `opencv-contrib-python` (no `opencv-python`) para tener soporte de ArUco.

---

## Sistema de control autónomo (`main/`)

### Arquitectura

El sistema corre tres hilos concurrentes más el display loop en el hilo principal:

```
Hilo principal  →  display loop (OpenCV imshow + matplotlib refresh)
Hilo 1          →  ArucoDetector.run()   — detección + solvePnP
Hilo 2          →  RCController.run()    — cálculo y envío de comandos RC
```

Un `threading.Lock` compartido protege el resultado de detección entre los hilos.

### Módulos

#### `main.py`
Punto de entrada. Instancia `TelloController` y llama a `ctrl.run()`.

```python
from tello_controller import TelloController

if __name__ == "__main__":
    ctrl = TelloController()
    ctrl.run()
```

---

#### `tello_controller.py` — `TelloController`

Orquesta todo el ciclo de vida:

| Método | Descripción |
|--------|-------------|
| `_connect()` | Conecta al Tello por Wi-Fi, activa el stream y reporta la batería |
| `_open_stream()` | Abre el stream UDP (`udp://@0.0.0.0:11111`), espera hasta 15 s a que estabilice |
| `run()` | Despega, lanza los hilos de detección y control, entra al display loop |
| `_display()` | Bucle principal: lee frame, llama al HUD, muestra ventana OpenCV, `q` para salir |

Al salir (`q`): detiene hilos, envía RC cero, aterriza, libera recursos.

---

#### `detector.py` — `ArucoDetector`

Detecta el marcador **ArUco ID=1** (diccionario `DICT_4X4_50`) y estima su pose 3D.

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `MARKER_SIZE` | `0.208` m | Lado real del marcador impreso |
| `DETECT_SCALE` | `0.5` | Escala de reducción del frame antes de detectar (más rápido) |

Flujo por frame:
1. Reduce el frame a la mitad para acelerar la detección.
2. Convierte a escala de grises y ejecuta `ArucoDetector.detectMarkers()`.
3. Si encuentra ID=1, escala las esquinas de vuelta a resolución original y llama a `cv2.solvePnP` (flag `SOLVEPNP_IPPE_SQUARE`).
4. Escribe en el lock compartido: `(corners, rvec, tvec, dist, cx, cy)`.

---

#### `controller.py` — `RCController`

Control proporcional que envía comandos RC al Tello cada 50 ms.

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `TARGET_DIST` | `1.2` m | Distancia objetivo al marcador |
| `DEAD_ZONE_D` | `0.08` m | Zona muerta en distancia |
| `Kp_dist` | `60` | Ganancia proporcional adelante/atrás |
| `MAX_VEL_FB` | `30` | Velocidad máxima adelante/atrás (%) |
| `DEAD_ZONE_PX` | `60` px | Zona muerta en píxeles para yaw y up/down |
| `MAX_VEL_YAW` | `60` | Velocidad máxima de rotación (%) |
| `MAX_VEL_UD` | `60` | Velocidad máxima arriba/abajo (%) |
| `LOST_TIMEOUT` | `0.5` s | Tiempo sin detección antes de detener el dron |

Canales RC:
- **Lateral (left/right):** siempre 0 (no usado).
- **Adelante/atrás:** proporcional al error `dist − TARGET_DIST`.
- **Arriba/abajo:** proporcional al error vertical en píxeles (`cy − CENTER_Y`).
- **Yaw:** proporcional al error horizontal en píxeles (`cx − CENTER_X`).

---

#### `hud.py` — `HUD` y `Plot3D`

**`HUD.draw(frame, res, lost)`** superpone sobre el frame de cámara:
- Contornos del marcador detectado (`drawDetectedMarkers`) y ejes 3D (`drawFrameAxes`).
- Cruz de crosshair y línea del centro al marcador.
- Distancia actual y estado (`OK` / `AVANZA` / `RETROCEDE`).
- Errores de píxel en X e Y.
- Coordenadas 3D del marcador en metros.
- Barra horizontal de error de distancia.

**`Plot3D`** abre una ventana matplotlib 3D (backend `TkAgg`) que muestra:
- El origen marcado como el ArUco.
- La posición actual del dron en rojo.
- Estela de las últimas `MAX_TRAIL = 80` posiciones.
- Línea punteada dron → ArUco.

Conversión de ejes de cámara a ejes intuitivos de la gráfica:

| Eje gráfica | Origen |
|-------------|--------|
| X (izq/der) | `tvec[0]` |
| Y (dist)    | `tvec[2]` |
| Z (arr/abj) | `-tvec[1]` |

---

#### `stream.py` — `LatestFrame`

Buffer de un solo frame que drena continuamente el buffer UDP del VideoCapture de OpenCV y conserva únicamente el frame más reciente, evitando acumulación de latencia.

```python
grabber = LatestFrame(cap)
frame   = grabber.frame   # propiedad thread-safe
```

---

### Ejecución del sistema principal

Desde la carpeta `main/files/`:

```bash
cd main/files
python main.py
```

| Tecla | Acción |
|-------|--------|
| `q`   | Aterrizar y salir |

> El marcador objetivo es **ArUco ID=1** (diccionario `DICT_4X4_50`). Asegúrate de que `aruco_id1.png` esté impreso con un lado real de **20.8 cm**.

---

## Flujo de calibración (fase previa)

### 0. Generar los patrones (solo una vez)

```bash
# Genera checkerboard_9x6.png (10×7 cuadros, 80 px/cuadro)
python checkerboard.py

# Genera marcadores ArUco
python arucos/arucocreation.py
```

Imprime `checkerboard_9x6.png` y mide el lado real de cada cuadro en metros — ese valor va en `SQUARE_SIZE` dentro de `arucos/calibracion.py`.

### 1. Capturar frames de calibración

```bash
python captureframes.py
```

| Tecla | Acción |
|-------|--------|
| `s`   | Guardar frame actual en `calib_frames/` |
| `q`   | Salir |

Captura **al menos 20–30 frames** variando inclinación, rotación y distancia del tablero.

### 2. Calibrar la cámara

```bash
python arucos/calibracion.py
```

Procesa todas las imágenes en `calib_frames/`, muestra las detecciones y guarda el resultado en `camera_params.npz`.

- RMS error < 1.0 px → calibración válida.

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
