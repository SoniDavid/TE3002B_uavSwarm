# 01_capture.py
import cv2, os
from djitellopy import Tello

SAVE_DIR = "calib_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

tello = Tello()
tello.connect()
print(f"Batería: {tello.get_battery()}%")
tello.streamon()
frame_read = tello.get_frame_read()

count = 0
print("'s' = guardar  |  'q' = salir")

while True:
    frame = frame_read.frame
    display = frame.copy()
    cv2.putText(display, f"Frames: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Tello", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        path = os.path.join(SAVE_DIR, f"frame_{count:03d}.png")
        cv2.imwrite(path, frame)
        print(f"  guardado {path}")
        count += 1
    elif key == ord('q'):
        break

tello.streamoff()
tello.end()
cv2.destroyAllWindows()
print(f"\nTotal capturado: {count} frames")