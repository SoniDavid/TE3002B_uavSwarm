"""
Tello Dual Camera Viewer
========================
Muestra en tiempo real:
  - Cámara frontal (color)
  - Cámara inferior (blanco y negro)

Controles:
  Q  →  Salir
  F  →  Cambiar a cámara frontal
  D  →  Cambiar a cámara inferior (downward)
"""

import cv2
from djitellopy import Tello

# ─── Configuración ────────────────────────────────────────────────────────────
WINDOW_NAME_FRONT = "Tello | Cámara Frontal"
WINDOW_NAME_DOWN  = "Tello | Cámara Inferior (B&N)"


def main():
    # Conectar al dron
    tello = Tello()
    tello.connect()

    print(f"✅ Conectado  |  Batería: {tello.get_battery()}%")

    # Iniciar stream de video
    tello.streamon()

    frame_reader = tello.get_frame_read()

    # Comenzar con cámara frontal
    tello.set_video_direction(Tello.CAMERA_FORWARD)
    current_cam = "front"

    print("Presiona  F = frontal | D = inferior | Q = salir")

    while True:
        # Leer frame actual
        frame = frame_reader.frame

        if frame is None:
            continue

        # ── Cámara frontal: mostrar en color ─────────────────────────────────
        if current_cam == "front":
            display = cv2.resize(frame, (960, 720))
            cv2.imshow(WINDOW_NAME_FRONT, display)
            # Ocultar la otra ventana si existe
            try:
                cv2.destroyWindow(WINDOW_NAME_DOWN)
            except Exception:
                pass

        # ── Cámara inferior: convertir a B&N ─────────────────────────────────
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = cv2.resize(gray, (960, 720))
            cv2.imshow(WINDOW_NAME_DOWN, display)
            try:
                cv2.destroyWindow(WINDOW_NAME_FRONT)
            except Exception:
                pass

        # ── Teclas ────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Saliendo...")
            break

        elif key == ord('f') and current_cam != "front":
            tello.set_video_direction(Tello.CAMERA_FORWARD)
            current_cam = "front"
            print("📷  Cámara FRONTAL")

        elif key == ord('d') and current_cam != "down":
            tello.set_video_direction(Tello.CAMERA_DOWNWARD)
            current_cam = "down"
            print("📷  Cámara INFERIOR (B&N)")

    # ── Limpieza ──────────────────────────────────────────────────────────────
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
    print("Conexión cerrada.")


if __name__ == "__main__":
    main()
