"""
Tello Dual Camera Viewer (versión optimizada)
=============================================
Controles:
  Q  →  Salir
  F  →  Cámara frontal
  D  →  Cámara inferior (B&N)
"""

import cv2
import time
from djitellopy import Tello

WINDOW_FRONT = "Tello | Frontal"
WINDOW_DOWN  = "Tello | Inferior (B&N)"
FPS_LIMIT    = 30  # máximo frames por segundo


def main():
    tello = Tello()
    tello.connect()
    print(f"✅ Conectado  |  Batería: {tello.get_battery()}%")

    tello.streamon()
    time.sleep(2)  # esperar a que el stream estabilice

    frame_reader = tello.get_frame_read()
    tello.set_video_direction(Tello.CAMERA_FORWARD)
    current_cam = "front"

    print("F = frontal | D = inferior | Q = salir")

    frame_time = 1.0 / FPS_LIMIT

    while True:
        start = time.time()

        frame = frame_reader.frame
        if frame is None:
            time.sleep(0.01)
            continue

        # Reducir resolución para aliviar carga
        frame = cv2.resize(frame, (640, 480))

        if current_cam == "front":
            cv2.imshow(WINDOW_FRONT, frame)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow(WINDOW_DOWN, gray)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('f') and current_cam != "front":
            tello.set_video_direction(Tello.CAMERA_FORWARD)
            current_cam = "front"
            cv2.destroyWindow(WINDOW_DOWN)
            print("📷 Cámara FRONTAL")
        elif key == ord('d') and current_cam != "down":
            tello.set_video_direction(Tello.CAMERA_DOWNWARD)
            current_cam = "down"
            cv2.destroyWindow(WINDOW_FRONT)
            print("📷 Cámara INFERIOR")

        # Limitar FPS para no saturar CPU
        elapsed = time.time() - start
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
    print("Conexión cerrada.")


if __name__ == "__main__":
    main()