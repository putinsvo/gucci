import sys
import json
import os
import cv2
import chess
import chess.engine
import numpy as np
import ctypes
from ctypes import wintypes
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QCheckBox
from PyQt6.QtCore import QTimer, Qt
from ultralytics import YOLO
import pydirectinput
import mss
import pygetwindow as gw

# --- НАСТРОЙКИ ---
CONFIG_FILE = "config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
else:
    cfg = {"window_title": "Roblox", "engine_path": "stockfish.exe", "auto_move": False}

# --- ЗАГРУЗКА МОДЕЛИ YOLO ---
model = YOLO("best.pt")

# --- ГЛОБАЛЬНЫЙ ХОТКЕЙ F12 (через Windows API) ---
user32 = ctypes.windll.user32
VK_F12 = 0x7B

class ChessBot(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess Overlay Control")
        self.setFixedSize(300, 200)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)

        self.auto_check = QCheckBox("Auto Move")
        self.auto_check.setChecked(cfg.get("auto_move", False))
        layout.addWidget(self.auto_check)

        self.stop_btn = QPushButton("STOP (F12)")
        self.stop_btn.clicked.connect(self.stop_bot)
        layout.addWidget(self.stop_btn)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000) # раз в секунду

        self.stopped = False

        # Таймер для проверки F12
        self.hotkey_timer = QTimer()
        self.hotkey_timer.timeout.connect(self.check_hotkey)
        self.hotkey_timer.start(100)

    def check_hotkey(self):
        # Если нажат F12 (0x8000 - флаг нажатия)
        if user32.GetAsyncKeyState(VK_F12) & 0x8000:
            self.stop_bot()

    def stop_bot(self):
        self.stopped = True
        self.auto_check.setChecked(False)
        self.status_label.setText("Status: STOPPED")

    def get_window_rect(self):
        try:
            wins = [w for w in gw.getWindowsWithTitle(cfg["window_title"]) if w.width > 100]
            if wins:
                w = wins[0]
                return w.left, w.top, w.width, w.height
        except:
            pass
        return None

    def tick(self):
        if self.stopped:
            return
        rect = self.get_window_rect()
        if not rect:
            self.status_label.setText("Status: Roblox window not found")
            return
        
        left, top, width, height = rect
        
        # Скриншот
        with mss.mss() as sct:
            monitor = {"top": top, "left": left, "width": width, "height": height}
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Распознавание
        from detect_custom import align_board, build_grid, grid_to_fen
        board_img = align_board(img)
        results = model.predict(board_img, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
        
        grid = build_grid(results)
        fen = grid_to_fen(grid, turn="w")
        self.status_label.setText(f"FEN: {fen[:20]}...")

        if self.auto_check.isChecked():
            # Stockfish
            try:
                engine = chess.engine.SimpleEngine.popen_uci(cfg["engine_path"])
                board = chess.Board(fen)
                result = engine.play(board, chess.engine.Limit(depth=12))
                move = result.move
                engine.quit()

                self.status_label.setText(f"Move: {move}")
                
                # Клик (пока на всё окно, потом поправим)
                file_map = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
                def uci_to_px(sq):
                    c = file_map[sq[0]]
                    r = 8 - int(sq[1])
                    cx = left + int((c + 0.5) * width / 8)
                    cy = top + int((r + 0.5) * height / 8)
                    return cx, cy

                fx, fy = uci_to_px(str(move)[:2])
                tx, ty = uci_to_px(str(move)[2:4])

                pydirectinput.moveTo(fx, fy)
                pydirectinput.click()
                pydirectinput.moveTo(tx, ty)
                pydirectinput.click()
            except Exception as e:
                self.status_label.setText(f"Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChessBot()
    window.show()
    sys.exit(app.exec())