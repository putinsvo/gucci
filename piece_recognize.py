import os
import cv2
import numpy as np

PIECES = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "assets", "templates")
CELL = 64

class PieceRecognizer:
    def __init__(self):
        self.templates: dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self):
        for name in PIECES:
            path = os.path.join(TEMPLATE_DIR, f"{name}.png")
            if os.path.isfile(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.templates[name] = self._extract_contour(img)

    @staticmethod
    def _extract_contour(img: np.ndarray) -> np.ndarray | None:
        """Извлекает бинарную маску контура фигуры."""
        img = cv2.resize(img, (CELL, CELL), interpolation=cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Фон клетки — это медиана углов
        corners = np.concatenate([img[:5, :5].flatten(), img[-5:, -5:].flatten()])
        bg_color = np.median(corners)
        
        # Выделяем всё, что не похоже на фон (фигуру)
        diff = cv2.absdiff(img, int(bg_color))
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Убираем шум
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Находим самый большой контур (это и есть фигура)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest_cnt = max(contours, key=cv2.contourArea)
        
        # Рисуем только этот контур на пустом холсте
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_cnt], -1, 255, -1)
        return clean_mask

    def classify_cell(self, cell_img: np.ndarray) -> str | None:
        if cell_img.size == 0:
            return None
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        mask = self._extract_contour(gray)
        
        if mask is None or np.count_nonzero(mask) < 50:
            return None

        best_name, best_score = None, 999.0
        for name, tpl in self.templates.items():
            if tpl is None: continue
            # matchShapes сравнивает геометрию контуров (0 = идеально, >1 = разные)
            score = cv2.matchShapes(mask, tpl, cv2.CONTOURS_MATCH_I1, 0.0)
            if score < best_score:
                best_score, best_name = score, name
                
        # Порог: если разница больше 0.15 — фигуры совсем не похожи
        return best_name if best_score < 0.15 else None

    def recognize_board(self, board_img: np.ndarray, orientation: str = "white") -> list[list[str | None]]:
        h, w = board_img.shape[:2]
        ch, cw = h / 8, w / 8
        board = [[None] * 8 for _ in range(8)]
        for r in range(8):
            for c in range(8):
                y0, y1 = int(r * ch), int((r + 1) * ch)
                x0, x1 = int(c * cw), int((c + 1) * cw)
                board[r][c] = self.classify_cell(board_img[y0:y1, x0:x1])
        if orientation == "black":
            board = [row[::-1] for row in board[::-1]]
        return board