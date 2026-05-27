# genera_aruco.py
import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_img = np.zeros((300, 300), dtype=np.uint8)
cv2.aruco.generateImageMarker(dictionary, 1, 300, marker_img)  # ID=1
cv2.imwrite("aruco_id1.png", marker_img)
print("Guardado aruco_id1.png — imprime y mide el lado en metros")