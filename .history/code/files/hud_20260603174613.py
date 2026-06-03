import cv2
import numpy as np

MARKER_SIZE = 0.208
TARGET_DIST = 1.2
DEAD_ZONE_D = 0.08
FRAME_W     = 960
FRAME_H     = 720
CENTER_X    = FRAME_W // 2
CENTER_Y    = FRAME_H // 2


class HUD:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs   = dist_coeffs

    def draw(self, frame, res, lost):
        if res is None:
            display = frame.copy()
            msg = "MARCADOR PERDIDO — detenido" if lost else "Buscando ArUco ID=1..."
            cv2.putText(display, msg, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 50, 255), 2)
            return display

        display = frame.copy()
        c, rvec, tvec, dist, cx, cy = res

        cv2.aruco.drawDetectedMarkers(display, [c])
        cv2.drawFrameAxes(display, self.camera_matrix, self.dist_coeffs,
                          rvec, tvec, MARKER_SIZE * 0.5)

        # Cruz central y línea al marcador
        cv2.line(display, (CENTER_X, 0),        (CENTER_X, FRAME_H), (255,255,255), 1)
        cv2.line(display, (0, CENTER_Y),         (FRAME_W,  CENTER_Y),(255,255,255), 1)
        cv2.circle(display, (cx, cy), 6, (0, 255, 255), -1)
        cv2.line(display, (CENTER_X, CENTER_Y), (cx, cy), (0, 255, 255), 1)

        error_d = dist - TARGET_DIST
        error_x = cx - CENTER_X
        error_y = cy - CENTER_Y
        status  = "OK" if abs(error_d) < DEAD_ZONE_D else \
                  ("RETROCEDE" if error_d < 0 else "AVANZA")
        color   = (0, 255, 0) if abs(error_d) < DEAD_ZONE_D else (0, 140, 255)

        cv2.putText(display, f"d={dist:.2f}m  {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display, f"ex={error_x:+d}px  ey={error_y:+d}px",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        # Barra de error de distancia
        bar_x = int(np.clip(CENTER_X + error_d * 120, 60, FRAME_W - 60))
        cv2.rectangle(display, (60, 8),       (FRAME_W - 60, 22), (60, 60, 60), -1)
        cv2.rectangle(display, (CENTER_X, 8), (bar_x, 22),         color,        -1)
        cv2.line(display, (CENTER_X, 4), (CENTER_X, 26), (255, 255, 255), 1)
        cv2.putText(display, f"Objetivo: {TARGET_DIST}m",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return display
