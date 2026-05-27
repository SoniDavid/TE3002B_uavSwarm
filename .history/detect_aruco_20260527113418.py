import cv2
import numpy as np
import time
from djitellopy import Tello

MARKER_SIZE = 0.05  # metros

data = np.load("camera_params.npz")
camera_matrix = data["K"]
dist_coeffs = data["dist"]

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, params)

# Conectar y activar stream
tello = Tello()
tello.connect()
print(f"Batería: {tello.get_battery()}%")
tello.streamon()
time.sleep(2)  # dar tiempo al Tello para empezar a enviar UDP

# Leer stream UDP directamente con OpenCV/FFMPEG (evita el crash de PyAV)
cap = cv2.VideoCapture("udp://@0.0.0.0:11111", cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimizar latencia

if not cap.isOpened():
    tello.streamoff()
    tello.end()
    raise RuntimeError("No se pudo abrir el stream UDP del Tello")

# Esperar frame con contenido real
print("Esperando stream estable...")
deadline = time.time() + 15
while time.time() < deadline:
    ret, frame = cap.read()
    if ret and frame is not None and frame.mean() > 2.0:
        print(f"Stream estable — {frame.shape[1]}x{frame.shape[0]}, "
              f"brillo: {frame.mean():.1f}")
        break
else:
    cap.release()
    tello.streamoff()
    tello.end()
    raise RuntimeError("Stream no estabilizó en 15 segundos")

print("Detectando ArUco ID=1 — presiona 'q' para salir")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            mask = (ids.flatten() == 1)
            corners_id1 = [corners[i] for i in range(len(ids)) if mask[i]]

            if corners_id1:
                cv2.aruco.drawDetectedMarkers(frame, corners_id1)

                for c in corners_id1:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        c, MARKER_SIZE, camera_matrix, dist_coeffs
                    )
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                      rvec[0], tvec[0], MARKER_SIZE * 0.5)
                    dist = float(np.linalg.norm(tvec[0]))
                    pos = tuple(c[0][0].astype(int))
                    cv2.putText(frame, f"ID=1  d={dist:.3f}m",
                                (pos[0], pos[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Bat: {tello.get_battery()}%",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("ArUco ID=1 - Tello", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
