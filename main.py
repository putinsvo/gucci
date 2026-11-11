import json
import os
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from capture import ScreenCapture
from board_detect import find_board_quad, warp_board, cell_center_in_frame
from piece_recognize import PieceRecognizer
from fen import board_to_fen, validate, side_to_move_from_diff, square_to_rc
from engine import UCIEngine
from overlay import Overlay
from mover import Mover
from ui import ControlPanel
from hotkeys import HotkeyManager

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config():
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


class App:
    def __init__(self, qapp):
        self.qapp = qapp
        self.cfg = load_config()
        self.capture = ScreenCapture(self.cfg.get("window_title", "Roblox"))
        self.recognizer = PieceRecognizer()
        self.engine = UCIEngine(self.cfg.get("engine_path", ""))
        self.hk = HotkeyManager(
            toggle_key=self.cfg.get("hotkey_toggle", "f8"),
            stop_key=self.cfg.get("hotkey_stop", "f12"),
        )
        self.mover = Mover(self.cfg.get("click_delay_ms", 80),
                           halt_flag=lambda: self.hk.stop_event.is_set())
        self.overlay = Overlay()
        self.panel = ControlPanel(self.cfg, on_toggle_auto=self._set_auto, on_stop=self._emergency_stop)
        self.hk.register()

        self.side = self.cfg.get("orientation", "white")
        self.auto_move = bool(self.cfg.get("auto_move", False))
        self.side_to_move = "w"  # переключается после каждого хода
        self.prev_board = None
        self.last_fen = ""
        self.engine_ready = False
        self._busy = False

        self.panel.show()
        self.overlay.show()

        self.timer = QTimer(); self.timer.timeout.connect(self._tick); self.timer.start(600)
        self.hk_timer = QTimer(); self.hk_timer.timeout.connect(self._poll_hotkeys); self.hk_timer.start(100)

    def _set_auto(self, v):
        self.auto_move = v
        self.panel.log_signal.emit(f"auto_move -> {v}")

    def _emergency_stop(self):
        self.hk.stop_event.set()
        self.auto_move = False
        self.panel.auto_check.setChecked(False)
        self.panel.log_signal.emit("EMERGENCY STOP")
        self.overlay.set_state("—", [], "—", "STOPPED")

    def _poll_hotkeys(self):
        if self.hk.toggle_event.is_set():
            self.hk.toggle_event.clear()
            self.auto_move = not self.auto_move
            self.panel.auto_check.setChecked(self.auto_move)
        if self.hk.stop_event.is_set() and self.auto_move:
            self.auto_move = False
            self.panel.auto_check.setChecked(False)

    def _ensure_engine(self):
        if self.engine_ready:
            return True
        try:
            self.engine.options.threads = int(self.cfg.get("threads", 4))
            self.engine.options.hash_mb = int(self.cfg.get("hash_mb", 256))
            self.engine.options.multipv = int(self.cfg.get("multipv", 3))
            self.engine.options.skill_level = int(self.cfg.get("skill_level", 20))
            self.engine.options.depth = int(self.cfg.get("depth", 18))
            self.engine.options.movetime_ms = int(self.cfg.get("movetime_ms", 1000))
            self.engine.start()
            self.engine_ready = True
            self.panel.log_signal.emit("engine ready")
            return True
        except Exception as e:
            self.panel.log_signal.emit(f"engine start failed: {e}")
            return False

    def _tick(self):
        if self._busy or self.hk.stop_event.is_set():
            return
        self._busy = True
        try:
            frame = self.capture.grab_window()
            if frame is None:
                return
            quad = find_board_quad(frame)
            if quad is None:
                self.overlay.set_state("—", [], "—", "no board")
                return
            board_img = warp_board(frame, quad)
            board = self.recognizer.recognize_board(board_img, self.side)

            # чей ход: если доска изменилась — обновляем
            if self.prev_board is not None:
                moved_side = side_to_move_from_diff(self.prev_board, board)
                if moved_side is not None:
                    self.side_to_move = moved_side
            self.prev_board = [row[:] for row in board]

            fen = board_to_fen(board, side_to_move=self.side_to_move)
            if not validate(fen):
                self.overlay.set_state("?", [], "", "invalid FEN")
                return
            if fen == self.last_fen:
                return
            self.last_fen = fen
            self.panel.log_signal.emit(f"FEN ({self.side_to_move}): {fen}")

            if not self._ensure_engine():
                return
            pv, bestmove = self.engine.analyze(fen)
            score = pv[0].score_str if pv else "—"
            status = "auto" if self.auto_move else "idle"
            self.overlay.set_state(score, pv, bestmove or "—", status)

            # ходим только когда наш ход и автопередвижение включено
            my_color = "w" if self.side == "white" else "b"
            if self.auto_move and self.side_to_move == my_color and bestmove and len(bestmove) >= 4:
                self._play_uci_move(bestmove, quad)
        except Exception as e:
            self.panel.log_signal.emit(f"tick error: {e}")
        finally:
            self._busy = False

    def _play_uci_move(self, uci, quad):
        win_x, win_y = self.capture.window_origin()
        from_rc = square_to_rc(uci[:2], self.side)
        to_rc = square_to_rc(uci[2:4], self.side)
        fx, fy = cell_center_in_frame(quad, *from_rc)
        tx, ty = cell_center_in_frame(quad, *to_rc)
        try:
            self.mover.make_move((win_x + fx, win_y + fy), (win_x + tx, win_y + ty))
            self.panel.log_signal.emit(f"played {uci}")
        except RuntimeError:
            self.panel.log_signal.emit("move aborted (halt)")

    def shutdown(self):
        self.hk.unregister()
        self.engine.stop()


def main():
    app = QApplication(sys.argv)
    a = App(app)
    app.aboutToQuit.connect(a.shutdown)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()