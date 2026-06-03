import cv2
import time
import threading
import numpy as np
from djitellopy import Tello

from stream     import LatestFrame
from detector   import ArucoDetector
from controller import RCController
from hud        import HUD

LOST_TIMEOUT = 0.5


class TelloController:
    def __init__(self):
        self.stop_event      = threading.Event()
        self.result_lock     = threading.Lock()
        self.detect_result   = None
        self.last_detect_time = 0.0

        # Cargar parámetros de cámara una sola vez y compartirlos
        data               = np.load("camera_params.npz")
        self.camera_matrix = data["K"]
        self.dist_coeffs   = data["dist"]

        self.detector = ArucoDetector(self.result_lock,
                                      self._set_result,
                                      self.stop_event)
        self.rc       = RCController(self.result_lock,
                                     self._get_result,
                                     self.stop_event)
        self.hud      = HUD(self.camera_matrix, self.dist_coeffs)

    def _set_result(self, result):
        self.detect_result = result
        if result is not None:
            self.last_detect_time = time.time()

    def _get_result(self):
        return self.detect_result, self.last_detect_time

    # ── Conexión ──────────────────────────────────────────────────────────────
    def _connect(self):
        self.tello = Tello()
        self.tello.connect()
        print(f"Batería: {self.tello.get_battery()}%")
        self.tello.streamon()
        time.sleep(2)

    # ── Stream ────────────────────────────────────────────────────────────────
    def _open_stream(self):
        cap = cv2.VideoCapture("udp://@0.0.0.0:11111", cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.tello.streamoff(); self.tello.end()
            raise RuntimeError("No se pudo abrir el stream UDP")

        self.grabber = LatestFrame(cap)
        self.cap     = cap

        print("Esperando stream estable...")
        deadline = time.time() + 15
        while time.time() < deadline:
            f = self.grabber.frame
            if f is not None and f.size > 0 and f.mean() > 2.0:
                print(f"Stream {f.shape[1]}x{f.shape[0]} — OK")
                return
            time.sleep(0.05)

        self.grabber.stop(); cap.release()
        self.tello.streamoff(); self.tello.end()
        raise RuntimeError("Stream no estabilizó en 15 s")

    # ── Display loop ──────────────────────────────────────────────────────────
    def _display(self):
        while True:
            frame = self.grabber.frame
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            with self.result_lock:
                res, last = self._get_result()

            lost    = (time.time() - last) > LOST_TIMEOUT
            display = self.hud.draw(frame, res, lost)

            cv2.imshow("Tello ArUco Control", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ── Entry point ───────────────────────────────────────────────────────────
    def run(self):
        self._connect()
        self._open_stream()
        self.tello.takeoff()
        time.sleep(2)

        threading.Thread(target=self.detector.run,
                         args=(self.grabber,), daemon=True).start()
        threading.Thread(target=self.rc.run,
                         args=(self.tello,),   daemon=True).start()

        print("Control activo — 'q' para aterrizar y salir")
        try:
            self._display()
        finally:
            self.stop_event.set()
            self.tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.3)
            self.tello.land()
            self.grabber.stop()
            self.cap.release()
            self.tello.streamoff()
            self.tello.end()
            cv2.destroyAllWindows()
