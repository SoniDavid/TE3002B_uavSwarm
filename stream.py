import cv2
import time
import threading


class LatestFrame:
    """Drena el buffer UDP y conserva solo el frame más reciente."""

    def __init__(self, cap):
        self.cap    = cap
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
                time.sleep(0.005)

    @property
    def frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._stop.set()
