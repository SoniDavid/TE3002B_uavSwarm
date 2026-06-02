import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "fflags;nobuffer|flags;low_delay|max_delay;0|"
    "probesize;32|analyzeduration;0|reorder_queue_size;0"
)
import cv2
import numpy as np
import time
import threading
from djitellopy import Tello
MARKER_SIZE  = 0.185
DETECT_SCALE = 0.4
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
# ── Estado compartido entre hilos ─────────────────────────────────────────────
latest_frame  = None   # frame más reciente (grabber → display)
detect_result = None   # metadatos de última detección (detect → display)
frame_lock    = threading.Lock()
result_lock   = threading.Lock()
stop_event    = threading.Event()
def grabber_thread(cap):
    global latest_frame
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret and frame is not None:
            with frame_lock:
                latest_frame = frame
def detection_thread():
    global detect_result
    last_id = None
    while not stop_event.is_set():
        with frame_lock:
            frame = latest_frame
        if frame is None or id(frame) == last_id:
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
                c = (corners_s[i] / DETECT_SCALE).astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_points, c[0], camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    result = (c, rvec, tvec, float(np.linalg.norm(tvec)))
                break   # solo ID=1
        with result_lock:
            detect_result = result
# ── Tello ─────────────────────────────────────────────────────────────────────
tello = Tello()
tello.connect()
print(f"Batería: {tello.get_battery()}%")
tello.streamon()
time.sleep(2)
cap = cv2.VideoCapture(
    "udp://@0.0.0.0:11111?fifo_size=0&overrun_nonfatal=1", cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
threading.Thread(target=grabber_thread, args=(cap,), daemon=True).start()
threading.Thread(target=detection_thread, daemon=True).start()
# Esperar stream estable
deadline = time.time() + 15
while time.time() < deadline:
    with frame_lock:
        f = latest_frame
    if f is not None and f.mean() > 2.0:
        print(f"Stream {f.shape[1]}x{f.shape[0]} — OK")
        break
    time.sleep(0.05)
else:
    stop_event.set(); cap.release(); tello.streamoff(); tello.end()
    raise RuntimeError("Stream no estabilizó")
print("Detectando ArUco ID=1 — 'q' para salir")
# ── Display loop ──────────────────────────────────────────────────────────────
try:
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        with result_lock:
            res = detect_result
        if res is not None:
            display = frame.copy()
            c, rvec, tvec, dist = res
            cv2.aruco.drawDetectedMarkers(display, [c])
            cv2.drawFrameAxes(display, camera_matrix, dist_coeffs,
                              rvec, tvec, MARKER_SIZE * 0.5)
            pos = tuple(c[0][0].astype(int))
            cv2.putText(display, f"ID=1  d={dist:.3f}m",
                        (pos[0], pos[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            display = frame
        cv2.imshow("Tello ArUco", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    stop_event.set()
    cap.release()
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
