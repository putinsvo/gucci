import os
import cv2

SRC = "board.png"
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "templates")
os.makedirs(OUT, exist_ok=True)

START = {
    (0, 0): "r", (0, 1): "n", (0, 2): "b", (0, 3): "q", (0, 4): "k", (0, 5): "b", (0, 6): "n", (0, 7): "r",
    (1, 0): "p", (1, 1): "p", (1, 2): "p", (1, 3): "p", (1, 4): "p", (1, 5): "p", (1, 6): "p", (1, 7): "p",
    (6, 0): "P", (6, 1): "P", (6, 2): "P", (6, 3): "P", (6, 4): "P", (6, 5): "P", (6, 6): "P", (6, 7): "P",
    (7, 0): "R", (7, 1): "N", (7, 2): "B", (7, 3): "Q", (7, 4): "K", (7, 5): "B", (7, 6): "N", (7, 7): "R",
}

img = cv2.imread(SRC)
if img is None:
    raise SystemExit(f"cannot read {SRC}")

# Обрезаем края с цифрами (3.5% с каждой стороны)
h, w = img.shape[:2]
margin_y = int(h * 0.035)
margin_x = int(w * 0.035)
img = img[margin_y:h-margin_y, margin_x:w-margin_x]

h, w = img.shape[:2]
ch, cw = h / 8, w / 8

for (r, c), name in START.items():
    y0, y1 = int(r * ch), int((r + 1) * ch)
    x0, x1 = int(c * cw), int((c + 1) * cw)
    
    # Берем ровно центр клетки, отступая 15% от краев, чтобы не захватить сетку
    cell_h, cell_w = y1 - y0, x1 - x0
    y0 += int(cell_h * 0.15)
    y1 -= int(cell_h * 0.15)
    x0 += int(cell_w * 0.15)
    x1 -= int(cell_w * 0.15)
    
    cell = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(OUT, f"{name}.png"), gray)
    print("wrote", name)