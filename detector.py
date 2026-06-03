import cv2
import time
import numpy as np

MARKER_SIZE  = 0.208
DETECT_SCALE = 0.5

OBJ_POINTS = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
], dtype=np.float32)


class ArucoDetector:
    def __init__(self, lock, set_result_fn, stop_event):
        data               = np.load("camera_params.npz")
        self.camera_matrix = data["K"]
        self.dist_coeffs   = data["dist"]
        self.lock          = lock
        self.set_result    = set_result_fn
        self.stop_event    = stop_event

        dictionary    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(
                            dictionary, cv2.aruco.DetectorParameters())

    def run(self, grabber):
        last_id = None

        while not self.stop_event.is_set():
            frame = grabber.frame
            if frame is None or id(frame) == last_id:
                time.sleep(0.002)
                continue
            last_id = id(frame)

            small = cv2.resize(frame, None,
                               fx=DETECT_SCALE, fy=DETECT_SCALE,
                               interpolation=cv2.INTER_NEAREST)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            corners_s, ids, _ = self.detector.detectMarkers(gray)

            result = None
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    if mid != 1:
                        continue
                    c  = (corners_s[i] / DETECT_SCALE).astype(np.float32)
                    ok, rvec, tvec = cv2.solvePnP(
                        OBJ_POINTS, c[0],
                        self.camera_matrix, self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        cx = int(c[0, :, 0].mean())
                        cy = int(c[0, :, 1].mean())
                        result = (c, rvec, tvec,
                                  float(np.linalg.norm(tvec)),
                                  cx, cy)
                    break

            with self.lock:
                self.set_result(result)
