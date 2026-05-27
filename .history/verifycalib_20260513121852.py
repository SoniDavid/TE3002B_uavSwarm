# 03_verify.py
import cv2, numpy as np
from djitellopy import Tello

data = np.load("camera_params.npz")
K, dist = data["K"], data["dist"]

tello = Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

print("Mostrando original vs undistorted. 'q' para salir.")
while True:
    frame = frame_read.frame
    undist = cv2.undistort(frame, K, dist)
    combined = np.hstack([frame, undist])
    cv2.putText(combined, "Original", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(combined, "Undistorted", (frame.shape[1]+10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Calibración", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
tello.end()
cv2.destroyAllWindows()