
import cv2
import numpy as np
import time
import threading
import queue
from djitellopy import Tello
# ── Configuración ──────────────────────────────────────────────────────────────
MARKER_SIZE    = 0.185          # metros
DISPLAY_FPS    = 30             # máximo FPS de display
DETECT_SCALE   = 0.5            # escala para detección
BAT_INTERVAL   = 5.0            # segundos entre consultas de batería
DISPLAY_MS     = max(1, int(1000 / DISPLAY_FPS))
data           = np.load("camera_params.npz")
camera_matrix  = data["K"]
dist_coeffs    = data["dist"]
# Matriz escalada para detección en resolución reducida
scaled_matrix  = camera_matrix.copy()
scaled_matrix[0, 0] *= DETECT_SCALE
scaled_matrix[1, 1] *= DETECT_SCALE
scaled_matrix[0, 2] *= DETECT_SCALE
scaled_matrix[1, 2] *= DETECT_SCALE
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params     = cv2.aruco.DetectorParameters()
detector   = cv2.aruco.ArucoDetector(dictionary, params)
obj_points = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
], dtype=np.float32)
# ── LatestFrame — drena el buffer y conserva solo el frame más reciente ────────
class LatestFrame:
    """
    Hilo daemon que consume continuamente el stream FFMPEG.
    Expone solo el último frame capturado sin bloquear al consumidor.
    Usa un Lock mínimo para intercambio atómico del puntero.
    """
    def __init__(self, cap):
        self.cap   = cap
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
    def _run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame      # intercambio de puntero (O(1))
            else:
                time.sleep(0.005)
    def get(self):
        with self._lock:
            return self._frame
    def stop(self):
        self._stop.set()
# ── DetectionThread — corre detección y pose estimation de forma asíncrona ─────
class DetectionThread:
    """
    Consume frames del LatestFrame, corre detección+solvePnP y publica
    el resultado (frame anotado + métricas) en self.result.
    El display solo lee self.result sin esperar al detector.
    """
    def __init__(self, grabber, camera_matrix, dist_coeffs):
        self.grabber        = grabber
        self.camera_matrix  = camera_matrix
        self.dist_coeffs    = dist_coeffs
        self._result        = None          # (annotated_frame, dist, pos) | None
        self._lock          = threading.Lock()
        self._stop          = threading.Event()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
    # ── Resultado publicado al display ─────────────────────────────────────────
    def get_result(self):
        with self._lock:
            return self._result
    def stop(self):
        self._stop.set()
    # ── Bucle del hilo de detección ────────────────────────────────────────────
    def _run(self):
        last_frame_id = id(None)    # evita reprocesar el mismo frame
        while not self._stop.is_set():
            frame = self.grabber.get()
            if frame is None or id(frame) == last_frame_id:
                time.sleep(0.002)   # espera activa mínima: ~0.002 s
                continue
            last_frame_id = id(frame)
            # Detección en resolución reducida
            small = cv2.resize(frame, None,
                               fx=DETECT_SCALE, fy=DETECT_SCALE,
                               interpolation=cv2.INTER_LINEAR)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            corners_s, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                with self._lock:
                    self._result = (frame, None, None)  # frame sin anotar
                continue
            # Escalar esquinas a resolución original
            corners_full = [c / DETECT_SCALE for c in corners_s]
            mask         = ids.flatten() == 1
            corners_id1  = [corners_full[i]
                            for i in range(len(ids)) if mask[i]]
            if not corners_id1:
                with self._lock:
                    self._result = (frame, None, None)
                continue
            # .copy() ocurre en el hilo de detección, no en el de display
            annotated = frame.copy()
            corners_np = [c.astype(np.float32) for c in corners_id1]
            cv2.aruco.drawDetectedMarkers(annotated, corners_np)
            dist_val = None
            pos_val  = None
            for c in corners_np:
                img_points = c[0]
                ok, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if not ok:
                    continue
                cv2.drawFrameAxes(annotated, self.camera_matrix,
                                  self.dist_coeffs, rvec, tvec,
                                  MARKER_SIZE * 0.5)
                dist_val = float(np.linalg.norm(tvec))
                pos_val  = tuple(c[0][0].astype(int))
                cv2.putText(annotated, f"ID=1  d={dist_val:.3f}m",
                            (pos_val[0], pos_val[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            with self._lock:
                self._result = (annotated, dist_val, pos_val)
# ── BatteryPoller — hilo daemon que no bloquea el display ─────────────────────
class BatteryPoller:
    def __init__(self, tello, interval=BAT_INTERVAL):
        self._tello    = tello
        self._interval = interval
        self._battery  = None
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
    def _run(self):
        while not self._stop.is_set():
            try:
                val = self._tello.get_battery()
                with self._lock:
                    self._battery = val
            except Exception:
                pass
            self._stop.wait(self._interval)   # interruptible
    def get(self):
        with self._lock:
            return self._battery
    def stop(self):
        self._stop.set()
# ── Tello: conexión y stream ───────────────────────────────────────────────────
tello = Tello()
tello.connect()
bat_poller = BatteryPoller(tello)
# Primera lectura síncrona para no mostrar "None" al inicio
bat_poller._battery = tello.get_battery()
print(f"Batería inicial: {bat_poller.get()}%")
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
detect = DetectionThread(grabber, camera_matrix, dist_coeffs)
print("Detectando ArUco ID=1 — presiona 'q' para salir")
# ── Loop de display — solo dibuja overlays baratos (batería) ───────────────────
try:
    while True:
        result = detect.get_result()
        if result is None:
            # Hilo de detección aún no ha publicado nada
            if cv2.waitKey(DISPLAY_MS) & 0xFF == ord("q"):
                break
            continue
        display, dist_val, pos_val = result
        # Overlay de batería: texto barato, no requiere copy()
        battery = bat_poller.get()
        bat_str = f"Bat: {battery}%" if battery is not None else "Bat: --"
        cv2.putText(display, bat_str,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("ArUco ID=1 - Tello", display)
        # waitKey mínimo para que OpenCV procese eventos de ventana
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    detect.stop()
    grabber.stop()
    bat_poller.stop()
    cap.release()
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
