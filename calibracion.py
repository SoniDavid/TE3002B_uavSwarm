# 02_calibrate.py
import cv2, numpy as np, glob

CHECKERBOARD = (9, 6)       # esquinas internas (no cuadros)
SQUARE_SIZE  = 0.025        # metros — ajusta según tu impresión real

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_pts, img_pts = [], []
images = sorted(glob.glob("calib_frames/*.png"))
print(f"{len(images)} imágenes encontradas")

valid = 0
for fname in images:
    img  = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        obj_pts.append(objp)
        img_pts.append(corners2)
        valid += 1
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Detección", img)
        cv2.waitKey(150)

cv2.destroyAllWindows()
print(f"Frames válidos: {valid}/{len(images)}")

if valid < 10:
    print("Necesitas más frames con variedad de poses.")
else:
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, gray.shape[::-1], None, None)
    print(f"\nRMS error: {rms:.4f} px  {'OK' if rms < 1.0 else '— considera recapturar'}")
    print(f"K:\n{K}")
    print(f"dist: {dist.ravel()}")
    np.savez("camera_params.npz", K=K, dist=dist)
    print("\ncamera_params.npz guardado")