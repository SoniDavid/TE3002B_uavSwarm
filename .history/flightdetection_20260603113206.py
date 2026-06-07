import cv2
import numpy as np
import time
import threading
from djitellopy import Tello

# ── Parámetros ArUco ──────────────────────────────────────────────────────────
MARKER_SIZE  = 0.208
DETECT_SCALE = 0.5

# ── Parámetros de control ─────────────────────────────────────────────────────
TARGET_DIST  = 1.2
DEAD_ZONE    = 0.08
Kp           = 60
MAX_VEL      = 30
LOST_TIMEOUT = 0.5

data          = np.load("camera_params.npz")
camera_matrix = data["K"]
dist_coeffs   = data["dist"]

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector   = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

obj_points = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0],
], dtype=np.float32)


# ── Grabber — drena el buffer UDP y guarda solo el frame más reciente ─────────
class LatestFrame:
    def __init__(self, cap):
        self.cap   = cap
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.005)   # evita busy-wait si el stream se pausa

    @property
    def frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._stop.set()


# ── Estado compartido detección ───────────────────────────────────────────────
detect_result    = None
last_detect_time = 0.0
result_lock      = threading.Lock()
stop_event       = threading.Event()


# ── Hilo de detección ─────────────────────────────────────────────────────────
def detection_thread(grabber):
    global detect_result, last_detect_time
    last_id = None

    while not stop_event.is_set():
        frame = grabber.frame
        if frame is None or id(frame) == last_id:
            time.sleep(0.002)
            continue
        last_id = id(frame)

        small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                           interpolation=cv2.INTER_NEAREST)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        corners_s, ids, _ = detector.detectMarkers(gray)

        result = None
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid != 1:
                    continue
                c  = (corners_s[i] / DETECT_SCALE).astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_points, c[0], camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    result = (c, rvec, tvec, float(np.linalg.norm(tvec)))
                break

        with result_lock:
            detect_result = result
            if result is not None:
                last_detect_time = time.time()


# ── Hilo de control RC ────────────────────────────────────────────────────────
def control_thread(tello):
    while not stop_event.is_set():
        with result_lock:
            res  = detect_result
            last = last_detect_time

        lost = (time.time() - last) > LOST_TIMEOUT

        if res is None or lost:
            tello.send_rc_control(0, 0, 0, 0)
        else:
            _, _, _, dist = res
            error = dist - TARGET_DIST
            if abs(error) < DEAD_ZONE:
                vel_fb = 0
            else:
                vel_fb = int(np.clip(Kp * error, -MAX_VEL, MAX_VEL))
            tello.send_rc_control(0, vel_fb, 0, 0)

        time.sleep(0.05)


# ── Conexión y stream ─────────────────────────────────────────────────────────
tello = Tello()
tello.connect()
print(f"Batería: {tello.get_battery()}%")
tello.streamon()
time.sleep(2)   # dar tiempo al Tello para levantar el stream UDP

# VideoCapture directo — más confiable que get_frame_read() para evitar frames negros
cap = cv2.VideoCapture("udp://@0.0.0.0:11111", cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # buffer mínimo → siempre el frame más reciente

if not cap.isOpened():
    tello.streamoff(); tello.end()
    raise RuntimeError("No se pudo abrir el stream UDP")

grabber = LatestFrame(cap)

# Esperar frame válido con contenido real (mean > 2 descarta frames negros)
print("Esperando stream estable...")
deadline = time.time() + 15
while time.time() < deadline:
    f = grabber.frame
    if f is not None and f.size > 0 and f.mean() > 2.0:
        print(f"Stream {f.shape[1]}x{f.shape[0]} — OK")
        break
    time.sleep(0.05)
else:
    grabber.stop(); cap.release()
    tello.streamoff(); tello.end()
    raise RuntimeError("Stream no estabilizó en 15 s")

tello.takeoff()
time.sleep(2)

threading.Thread(target=detection_thread, args=(grabber,), daemon=True).start()
threading.Thread(target=control_thread,   args=(tello,),   daemon=True).start()
print("Control activo — 'q' para aterrizar y salir")


# ── Display loop ──────────────────────────────────────────────────────────────
try:
    while True:
        frame = grabber.frame
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        with result_lock:
            res  = detect_result
            last = last_detect_time

        lost = (time.time() - last) > LOST_TIMEOUT

        if res is not None:
            display = frame.copy()
            c, rvec, tvec, dist = res

            cv2.aruco.drawDetectedMarkers(display, [c])
            cv2.drawFrameAxes(display, camera_matrix, dist_coeffs,
                              rvec, tvec, MARKER_SIZE * 0.5)

            error  = dist - TARGET_DIST
            status = "OK" if abs(error) < DEAD_ZONE else ("RETROCEDE" if error < 0 else "AVANZA")
            color  = (0, 255, 0) if abs(error) < DEAD_ZONE else (0, 140, 255)

            pos = tuple(c[0][0].astype(int))
            cv2.putText(display, f"d={dist:.2f}m  err={error:+.2f}m  {status}",
                        (pos[0], pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Barra de error visual
            bar_x = int(np.clip(320 + error * 120, 60, 580))
            cv2.rectangle(display, (60, 8),   (580, 22), (60, 60, 60), -1)
            cv2.rectangle(display, (320, 8),  (bar_x, 22), color, -1)
            cv2.line(display, (320, 4), (320, 26), (255, 255, 255), 1)
            cv2.putText(display, f"Objetivo: {TARGET_DIST}m",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            display = frame.copy()
            msg = "MARCADOR PERDIDO — detenido" if lost else "Buscando ArUco ID=1..."
            cv2.putText(display, msg, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 50, 255), 2)

        cv2.imshow("Tello ArUco Control", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    stop_event.set()
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(0.3)
    tello.land()
    grabber.stop()
    cap.release()
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()