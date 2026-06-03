import cv2
import numpy as np
import threading
import matplotlib
matplotlib.use("TkAgg")          # backend no-bloqueante fuera del hilo principal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

MARKER_SIZE = 0.208
TARGET_DIST = 1.2
DEAD_ZONE_D = 0.08
FRAME_W     = 960
FRAME_H     = 720
CENTER_X    = FRAME_W // 2
CENTER_Y    = FRAME_H // 2

# Historial de posiciones del dron (últimos N puntos)
MAX_TRAIL   = 80


class Plot3D:
    """Ventana matplotlib 3D que muestra la posición del dron relativa al ArUco."""

    def __init__(self):
        self._lock   = threading.Lock()
        self._tvec   = None          # posición actual del dron
        self._trail  = []            # historial de posiciones

        self._fig = plt.figure("Posición 3D", figsize=(5, 5))
        self._ax  = self._fig.add_subplot(111, projection="3d")
        self._fig.patch.set_facecolor("#1a1a2e")
        self._ax.set_facecolor("#1a1a2e")

        self._setup_axes()
        plt.ion()
        plt.show()

    def _setup_axes(self):
        ax = self._ax
        ax.set_xlabel("X  (izq/der)",  color="white", labelpad=6)
        ax.set_ylabel("Z  (dist)",     color="white", labelpad=6)
        ax.set_zlabel("Y  (arr/abj)",  color="white", labelpad=6)
        ax.tick_params(colors="white")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#444")
        ax.grid(True, color="#333")

    def update(self, tvec):
        """Llamado desde el hilo de detección con el tvec actual."""
        # tvec viene en coordenadas de cámara: X=der, Y=abj, Z=adelante
        # Remapeamos a ejes intuitivos para la gráfica:
        #   plot_X =  tvec[0]   (izquierda/derecha)
        #   plot_Y =  tvec[2]   (distancia al frente)
        #   plot_Z = -tvec[1]   (arriba/abajo — invertido porque Y cámara apunta abajo)
        x =  float(tvec[0])
        y =  float(tvec[2])
        z = -float(tvec[1])
        with self._lock:
            self._tvec = (x, y, z)
            self._trail.append((x, y, z))
            if len(self._trail) > MAX_TRAIL:
                self._trail.pop(0)

    def refresh(self):
        """Llamado desde el display loop (hilo principal) para redibujar."""
        with self._lock:
            tvec  = self._tvec
            trail = list(self._trail)

        self._ax.cla()
        self._setup_axes()

        lim = 2.5   # metros — rango de los ejes
        self._ax.set_xlim(-lim, lim)
        self._ax.set_ylim(0, lim * 2)
        self._ax.set_zlim(-lim, lim)
        self._ax.set_title("Dron relativo al ArUco", color="white", pad=8)

        # Origen = ArUco
        self._ax.scatter(0, 0, 0,
                         color="#00ffcc", s=120, zorder=5,
                         label="ArUco (origen)")
        self._ax.text(0, 0, 0, "  ArUco", color="#00ffcc", fontsize=8)

        if tvec is not None:
            x, y, z = tvec

            # Estela de posiciones pasadas
            if len(trail) > 1:
                tx, ty, tz = zip(*trail)
                self._ax.plot(tx, ty, tz,
                              color="#5599ff", linewidth=0.8, alpha=0.5)

            # Posición actual del dron
            self._ax.scatter(x, y, z,
                             color="#ff4444", s=160, zorder=5,
                             label="Dron")
            self._ax.text(x, y, z,
                          f"  ({x:.2f}, {y:.2f}, {z:.2f})m",
                          color="#ff8888", fontsize=7)

            # Línea punteada dron → ArUco
            self._ax.plot([0, x], [0, y], [0, z],
                          color="#888888", linestyle="--", linewidth=0.8)

        self._ax.legend(loc="upper left",
                        facecolor="#1a1a2e", edgecolor="#444",
                        labelcolor="white", fontsize=8)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


class HUD:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs   = dist_coeffs
        self.plot3d        = Plot3D()

    def draw(self, frame, res, lost):
        # ── Actualizar gráfica 3D ─────────────────────────────────────────────
        if res is not None:
            _, _, tvec, _, _, _ = res
            self.plot3d.update(tvec)
        self.plot3d.refresh()

        # ── Frame de cámara ───────────────────────────────────────────────────
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

        # Coordenadas 3D en pantalla
        x =  float(tvec[0]);  y = float(tvec[2]);  z = -float(tvec[1])
        cv2.putText(display, f"3D  X={x:+.2f}  Y={y:.2f}  Z={z:+.2f} m",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

        # Barra de error de distancia
        bar_x = int(np.clip(CENTER_X + error_d * 120, 60, FRAME_W - 60))
        cv2.rectangle(display, (60, 8),       (FRAME_W - 60, 22), (60, 60, 60), -1)
        cv2.rectangle(display, (CENTER_X, 8), (bar_x, 22),         color,        -1)
        cv2.line(display, (CENTER_X, 4), (CENTER_X, 26), (255, 255, 255), 1)
        cv2.putText(display, f"Objetivo: {TARGET_DIST}m",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return display