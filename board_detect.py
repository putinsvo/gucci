import cv2
import numpy as np


def find_board_quad(frame: np.ndarray) -> np.ndarray | None:
    """Возвращает упорядоченные 4 угла доски в координатах кадра (TL, TR, BR, BL)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = frame.shape[:2]
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h_img * w_img) * 0.05:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best, best_area = approx, area
    if best is None:
        return None

    pts = best.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    ordered = np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)],   # BL
    ], dtype="float32")
    return ordered


def warp_board(frame: np.ndarray, quad: np.ndarray, size: int = 640) -> np.ndarray:
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(frame, M, (size, size))


def cell_center_in_frame(quad: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Центр клетки (r,c) в координатах кадра. r=0 — верх доски, c=0 — левый столбец."""
    size = 8.0
    u = (c + 0.5) / size
    v = (r + 0.5) / size
    tl, tr, br, bl = quad
    top = tl * (1 - u) + tr * u
    bot = bl * (1 - u) + br * u
    pt = top * (1 - v) + bot * v
    return int(round(pt[0])), int(round(pt[1]))