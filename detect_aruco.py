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

# ── Estado compartido ─────────────────────────────────────────────────────────
detect_result = None   # (corners, rvec, tvec, dist) | None
result_lock   = threading.Lock()
stop_event    = threading.Event()


def detection_thread(frame_read):
    global detect_result
    last_id = None

    while not stop_event.is_set():
        frame = frame_read.frame
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
                c = (corners_s[i] / DETECT_SCALE).astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_points, c[0], camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    result = (c, rvec, tvec, float(np.linalg.norm(tvec)))
                break

        with result_lock:
            detect_result = result


# ── Tello ─────────────────────────────────────────────────────────────────────
tello = Tello()
tello.connect()
print(f"Batería: {tello.get_battery()}%")
tello.streamon()
time.sleep(2)

frame_read = tello.get_frame_read()

# Esperar primer frame válido
deadline = time.time() + 15
while time.time() < deadline:
    f = frame_read.frame
    if f is not None and f.size > 0 and f.mean() > 2.0:
        print(f"Stream {f.shape[1]}x{f.shape[0]} — OK")
        break
    time.sleep(0.05)
else:
    tello.streamoff(); tello.end()
    raise RuntimeError("Stream no estabilizó")

threading.Thread(target=detection_thread, args=(frame_read,), daemon=True).start()
print("Detectando ArUco ID=1 — 'q' para salir")

# ── Display loop ──────────────────────────────────────────────────────────────
try:
    while True:
        frame = frame_read.frame
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            display = frame

        cv2.imshow("Tello ArUco", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    stop_event.set()
    frame_read.stop()
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
