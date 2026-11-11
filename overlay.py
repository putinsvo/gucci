from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF
from PyQt6.QtWidgets import QWidget

from engine import PVLine


class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.resize(500, 260)
        self.move(30, 30)

        self.score_text = "—"
        self.pv_lines: list[PVLine] = []
        self.best_move_text = "—"
        self.status_text = "idle"

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(100)

    def set_state(self, score_text: str, pv_lines: list[PVLine], best_move: str, status: str):
        self.score_text = score_text
        self.pv_lines = pv_lines
        self.best_move_text = best_move
        self.status_text = status
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg = QRectF(0, 0, self.width(), self.height())
        p.setBrush(QColor(20, 20, 24, 200))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(bg, 12, 12)

        p.setPen(QPen(QColor(230, 230, 240)))
        f = QFont("Consolas", 11)
        p.setFont(f)

        y = 22
        p.drawText(14, y, f"Eval: {self.score_text}    Best: {self.best_move_text}    [{self.status_text}]")
        y += 22

        for line in self.pv_lines:
            p.drawText(14, y, f"#{line.multipv} d{line.depth} {line.score_str}  {' '.join(line.pv[:6])}")
            y += 20

        # стрелка хода на доске
        if self.best_move_text and len(self.best_move_text) >= 4:
            self._draw_arrow(p)
        p.end()

    def _draw_arrow(self, p: QPainter):
        from fen import square_to_rc
        from_sq, to_sq = self.best_move_text[:2], self.best_move_text[2:4]
        # оверлей растягивается под окно Roblox; для наглядности — рисуем в относительных координатах
        # реальные пиксели подставит main.py, если знает bbox доски
        pass