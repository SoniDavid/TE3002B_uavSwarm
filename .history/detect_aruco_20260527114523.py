import cv2
import numpy as np
import time
import threading
from djitellopy import Tello

# ── Configuración ──────────────────────────────────────────────────────────────
MARKER_SIZE   = 0.185   # metros
TARGET_FPS    = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS
BAT_INTERVAL  = 5.0     # segundos entre consultas de batería
DETECT_SCALE  = 0.5     # escala para detección (display sigue en full res)

data          = np.load("camera_params.npz")
camera_matrix = data["K"]
dist_coeffs   = data["dist"]

# Matriz escalada para detección en resolución reducida
scaled_matrix = camera_matrix.copy()
scaled_matrix[0, 0] *= DETECT_SCALE   # fx
scaled_matrix[1, 1] *= DETECT_SCALE   # fy
scaled_matrix[0, 2] *= DETECT_SCALE   # cx
scaled_matrix[1, 2] *= DETECT_SCALE   # cy

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params     = cv2.aruco.DetectorParameters()
detector   = cv2.aruco.ArucoDetector(dictionary, params)

# Objeto para estimación de pose (reemplaza la API deprecada)
obj_points = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
], dtype=np.float32)


# ── Grabber ────────────────────────────────────────────────────────────────────
class LatestFrame:
    """Drena el buffer de FFMPEG y conserva solo el frame más reciente."""

    def __init__(self, cap):
        self.cap    = cap
        self.frame  = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.frame = frame
            else:
                # Evita busy-wait cuando el stream no entrega frames
                time.sleep(0.005)

    def get(self):
        with self._lock:
            return self.frame

    def stop(self):
        self._stop.set()


# ── Tello ──────────────────────────────────────────────────────────────────────
tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"Batería: {battery}%")
tello.streamon()
time.sleep(2)

cap = cv2.VideoCapture("udp://@0.0.0.0:11111", cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    tello.streamoff(); tello.end()
    raise RuntimeError("No se pudo abrir el stream UDP del Tello")

grabber = LatestFrame(cap)

print("Esperando stream estable...")
deadline = time.time() + 15
while time.time() < deadline:
    frame = grabber.get()
    if frame is not None and frame.mean() > 2.0:
        h, w = frame.shape[:2]
        print(f"Stream estable — {w}x{h}")
        break
    time.sleep(0.05)
else:
    grabber.stop(); cap.release()
    tello.streamoff(); tello.end()
    raise RuntimeError("Stream no estabilizó en 15 segundos")

print("Detectando ArUco ID=1 — presiona 'q' para salir")

# ── Loop principal ─────────────────────────────────────────────────────────────
last_bat_time = time.time()

try:
    while True:
        t0 = time.time()

        frame = grabber.get()
        if frame is None:
            time.sleep(0.005)
            continue

        # ── Batería: consultar solo cada BAT_INTERVAL segundos ──
        now = time.time()
        if now - last_bat_time >= BAT_INTERVAL:
            battery       = tello.get_battery()
            last_bat_time = now

        # ── Detección en resolución reducida ──
        small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                           interpolation=cv2.INTER_LINEAR)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        corners_s, ids, _ = detector.detectMarkers(gray)

        display = frame   # sin .copy() — solo escribimos encima

        if ids is not None:
            # Escalar esquinas de vuelta a resolución original
            corners_full = [c / DETECT_SCALE for c in corners_s]
            mask         = ids.flatten() == 1
            corners_id1  = [corners_full[i] for i in range(len(ids)) if mask[i]]

            if corners_id1:
                display = frame.copy()   # .copy() solo cuando vamos a dibujar
                corners_id1_np = [c.astype(np.float32) for c in corners_id1]
                cv2.aruco.drawDetectedMarkers(display, corners_id1_np)

                for c in corners_id1_np:
                    # API moderna: solvePnP en lugar de estimatePoseSingleMarkers
                    img_points = c[0]
                    ok, rvec, tvec = cv2.solvePnP(
                        obj_points, img_points,
                        camera_matrix, dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if not ok:
                        continue

                    cv2.drawFrameAxes(display, camera_matrix, dist_coeffs,
                                      rvec, tvec, MARKER_SIZE * 0.5)

                    dist = float(np.linalg.norm(tvec))
                    pos  = tuple(c[0][0].astype(int))
                    cv2.putText(display, f"ID=1  d={dist:.3f}m",
                                (pos[0], pos[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display, f"Bat: {battery}%",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("ArUco ID=1 - Tello", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # ── Limitar FPS para no saturar el hilo de detección ──
        elapsed = time.time() - t0
        if elapsed < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - elapsed)

finally:
    grabber.stop()
    cap.release()
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()