# genera_checkerboard.py
import cv2
import numpy as np

cols, rows = 10, 7          # cuadros (esquinas internas = cols-1, rows-1)
square_px  = 80             # píxeles por cuadro
h = rows * square_px
w = cols * square_px

board = np.zeros((h, w), dtype=np.uint8)
for r in range(rows):
    for c in range(cols):
        if (r + c) % 2 == 0:
            y1, y2 = r*square_px, (r+1)*square_px
            x1, x2 = c*square_px, (c+1)*square_px
            board[y1:y2, x1:x2] = 255

cv2.imwrite("checkerboard_9x6.png", board)
print("Guardado checkerboard_9x6.png")