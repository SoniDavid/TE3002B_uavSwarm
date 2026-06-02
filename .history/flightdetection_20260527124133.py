"""
flightdetection.py — Seguimiento autónomo de ArUco ID=1 con DJI Tello.

Basado en:
  - detect_aruco.py (detección funcional)
  - green tracker   (control de vuelo funcional)

Control:
  - yaw           → centrar marcador horizontalmente
  - up/down       → centrar marcador verticalmente
  - forward/back  → mantener distancia a 50 cm
  - left/right    → 0 (no se usa)

Seguridad:
  - Sin marcador  → hover
  - Batería < 15% → aterrizar
  - Tecla 'q'     → aterrizar
  - Tecla 'e'     → emergency (sin aterrizaje controlado)
"""

import cv2
import numpy as np
import time
import threading
from djitellopy import Tello

# ── Configuración ─────────────────────────────────────────────────────────────
MARKER_SIZE    = 0.185       # metros
TARGET_DIST    = 0.50        # metros — distancia deseada al marcador
DETECT_SCALE   = 0.5         # escala para detección

FRAME_W        = 480         # resolución de display
FRAME_H        = 360
CENTER_X       = FRAME_W // 2  # 240
CENTER_Y       = FRAME_H // 2  # 180

DEAD_ZONE_PX   = 30          # pixeles — sin reacción (yaw / ud)
DEAD_ZONE_DIST = 0.05        # metros  — sin reacción (fb)
MAX_SPEED      = 35          # velocidad máxima (rango Tello: -100..100)
BAT_MIN        = 15          # % mínimo antes de auto-land
MAX_TIME       = 120         # segundos máximos de vuelo

# ── Calibración de cámara ─────────────────────────────────────────────────────
data          = np.load("camera_params.npz")
camera_matrix = data["K"]
dist_coeffs   = data["dist"]

# ── ArUco ─────────────────────────────────────────────────────────────────────
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector   = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

obj_points = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
], dtype=np.float32)

# ── Estado compartido ─────────────────────────────────────────────────────────
detect_result = None   # (corners_full, rvec, tvec, dist_m) | None
result_lock   = threading.Lock()
stop_event    = threading.Event()


# ── Hilo de detección (idéntico a detect_aruco.py) ────────────────────────────
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


# ── Tello: conectar y despegar ────────────────────────────────────────────────
tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"Batería: {battery}%")

if battery < BAT_MIN:
    print(f"Batería demasiado baja. Abortando.")
    tello.end()
    exit()

# Despegar PRIMERO, luego stream (patrón que funciona en vuelo)
tello.takeoff()
time.sleep(1)

tello.streamon()
time.sleep(1)
frame_read = tello.get_frame_read()

# Esperar stream estable
print("Esperando stream...")
deadline = time.time() + 15
while time.time() < deadline:
    f = frame_read.frame
    if f is not None and f.size > 0 and f.mean() > 2.0:
        orig_h, orig_w = f.shape[:2]
        print(f"Stream {orig_w}x{orig_h} — OK")
        break
    time.sleep(0.05)
else:
    tello.land()
    tello.streamoff()
    tello.end()
    raise RuntimeError("Stream no estabilizó")

# Factores de escala: resolución original → display
scale_x = FRAME_W / orig_w
scale_y = FRAME_H / orig_h

# Iniciar detección
threading.Thread(target=detection_thread, args=(frame_read,), daemon=True).start()

start_time = time.time()
emergency_abort = False
print(f"Tracking ArUco ID=1 — target={TARGET_DIST}m | 'q'=aterrizar | 'e'=emergencia")

# ── Loop principal (mismo patrón que green tracker) ───────────────────────────
try:
    while True:
        elapsed = time.time() - start_time

        frame = frame_read.frame
        if frame is None:
            continue

        # Resize crea un array nuevo — seguro para dibujar encima
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        with result_lock:
            res = detect_result

        yaw_speed = 0
        ud_speed  = 0
        fb_speed  = 0

        if res is not None:
            corners_full, rvec, tvec, dist_m = res

            # Centro del marcador escalado a resolución de display
            center_orig = corners_full[0].mean(axis=0)
            cx = int(center_orig[0] * scale_x)
            cy = int(center_orig[1] * scale_y)

            # ── Eje horizontal → yaw (girar hacia el marcador) ────────────
            error_x = cx - CENTER_X   # positivo = marcador a la derecha
            if abs(error_x) > DEAD_ZONE_PX:
                speed = int(np.interp(abs(error_x),
                                      [DEAD_ZONE_PX, CENTER_X],
                                      [15, MAX_SPEED]))
                yaw_speed = speed if error_x > 0 else -speed

            # ── Eje vertical → subir / bajar ──────────────────────────────
            error_y = cy - CENTER_Y   # positivo = marcador abajo
            if abs(error_y) > DEAD_ZONE_PX:
                speed = int(np.interp(abs(error_y),
                                      [DEAD_ZONE_PX, CENTER_Y],
                                      [15, MAX_SPEED]))
                ud_speed = -speed if error_y > 0 else speed

            # ── Distancia → forward / back ────────────────────────────────
            error_dist = dist_m - TARGET_DIST  # positivo = muy lejos
            if abs(error_dist) > DEAD_ZONE_DIST:
                speed = int(np.interp(abs(error_dist),
                                      [DEAD_ZONE_DIST, 1.0],
                                      [15, MAX_SPEED]))
                fb_speed = speed if error_dist > 0 else -speed

            # ── Dibujar overlay ───────────────────────────────────────────
            # Esquinas escaladas para dibujar el contorno
            corners_disp = (corners_full[0] * np.array([scale_x, scale_y])).astype(int)
            for j in range(4):
                pt1 = tuple(corners_disp[j])
                pt2 = tuple(corners_disp[(j + 1) % 4])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.circle(frame, (CENTER_X, CENTER_Y), 5, (255, 255, 255), -1)

            # Dirección visual
            h_dir = "→R" if yaw_speed > 0 else ("←L" if yaw_speed < 0 else "·")
            v_dir = "↓"  if ud_speed < 0  else ("↑"  if ud_speed > 0  else "·")
            f_dir = "→FW" if fb_speed > 0 else ("←BK" if fb_speed < 0 else "·")

            cv2.putText(frame,
                        f"ID=1 d={dist_m:.2f}m | {h_dir} yaw={yaw_speed:+d} | "
                        f"{v_dir} ud={ud_speed:+d} | {f_dir} fb={fb_speed:+d}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Center:({cx},{cy}) ex={error_x:+d} ey={error_y:+d} "
                        f"ed={error_dist:+.2f}m",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            print(f"[{elapsed:5.1f}s] ID=1 d={dist_m:.2f}m | "
                  f"ex={error_x:+d} yaw={yaw_speed:+d} | "
                  f"ey={error_y:+d} ud={ud_speed:+d} | "
                  f"ed={error_dist:+.2f} fb={fb_speed:+d}")

        else:
            cv2.putText(frame, "No ArUco - hovering",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # lr=0 siempre — solo yaw, fb, ud
        tello.send_rc_control(0, fb_speed, ud_speed, yaw_speed)

        # HUD
        remaining = max(0.0, MAX_TIME - elapsed)
        cv2.putText(frame, f"Bat:{battery}% | Time:{remaining:.0f}s",
                    (10, FRAME_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Tello ArUco Tracking", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or elapsed >= MAX_TIME:
            reason = "'q'" if key == ord('q') else "tiempo máximo"
            print(f"Fin ({reason}) — aterrizando...")
            break

        if key == ord('e'):
            print("EMERGENCIA — sin aterrizaje controlado")
            emergency_abort = True
            break

except KeyboardInterrupt:
    print("Interrupción manual — aterrizando...")

finally:
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(0.3)
    stop_event.set()
    if not emergency_abort:
        tello.land()
    frame_read.stop()
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
    print("Recursos liberados.")
