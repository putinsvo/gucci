import json
import os
import threading

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QFileDialog, QSpinBox, QComboBox, QCheckBox, QTextEdit,
)

from engine import UCIEngine


class ControlPanel(QWidget):
    log_signal = pyqtSignal(str)

    def __init__(self, config: dict, on_toggle_auto, on_stop):
        super().__init__()
        self.config = config
        self.on_toggle_auto = on_toggle_auto
        self.on_stop = on_stop
        self.setWindowTitle("Chess Overlay — Control")
        self.setFixedSize(520, 560)

        layout = QVBoxLayout(self)

        # engine path
        row = QHBoxLayout()
        row.addWidget(QLabel("Engine .exe:"))
        self.engine_path_edit = QLineEdit(config.get("engine_path", ""))
        row.addWidget(self.engine_path_edit)
        btn = QPushButton("...")
        btn.clicked.connect(self._pick_engine)
        row.addWidget(btn)
        layout.addLayout(row)

        # window title
        row = QHBoxLayout()
        row.addWidget(QLabel("Window title:"))
        self.window_edit = QLineEdit(config.get("window_title", "Roblox"))
        row.addWidget(self.window_edit)
        layout.addLayout(row)

        # side
        row = QHBoxLayout()
        row.addWidget(QLabel("Side:"))
        self.side_combo = QComboBox()
        self.side_combo.addItems(["white", "black"])
        self.side_combo.setCurrentText(config.get("orientation", "white"))
        row.addWidget(self.side_combo)
        layout.addLayout(row)

        # params
        self.threads_spin = self._spin(layout, "Threads", 1, 64, config.get("threads", 4))
        self.hash_spin = self._spin(layout, "Hash MB", 16, 4096, config.get("hash_mb", 256))
        self.multipv_spin = self._spin(layout, "MultiPV", 1, 5, config.get("multipv", 3))
        self.depth_spin = self._spin(layout, "Depth", 1, 40, config.get("depth", 18))
        self.movetime_spin = self._spin(layout, "Movetime ms", 0, 30000, config.get("movetime_ms", 1000))
        self.skill_spin = self._spin(layout, "Skill Level", 0, 20, config.get("skill_level", 20))
        self.delay_spin = self._spin(layout, "Click delay ms", 10, 1000, config.get("click_delay_ms", 80))

        # auto move
        self.auto_check = QCheckBox("Auto move (F8 toggles)")
        self.auto_check.setChecked(bool(config.get("auto_move", False)))
        self.auto_check.stateChanged.connect(self._on_auto_changed)
        layout.addWidget(self.auto_check)

        # buttons
        btns = QHBoxLayout()
        self.start_btn = QPushButton("Start engine")
        self.start_btn.clicked.connect(self._start_engine)
        btns.addWidget(self.start_btn)
        self.stop_btn = QPushButton("EMERGENCY STOP (F12)")
        self.stop_btn.setStyleSheet("background:#b00020;color:white;font-weight:bold;")
        self.stop_btn.clicked.connect(self.on_stop)
        btns.addWidget(self.stop_btn)
        layout.addLayout(btns)

        # log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.log_signal.connect(self.log.append)

    def _spin(self, layout, label, lo, hi, val):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(int(val))
        row.addWidget(sp)
        layout.addLayout(row)
        return sp

    def _pick_engine(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select engine", "", "Executables (*.exe);;All files (*)")
        if path:
            self.engine_path_edit.setText(path)

    def _on_auto_changed(self, state):
        self.config["auto_move"] = bool(state)
        self.on_toggle_auto(bool(state))
        self.log_signal.emit(f"auto_move = {self.config['auto_move']}")

    def _start_engine(self):
        self.config["engine_path"] = self.engine_path_edit.text()
        self.config["window_title"] = self.window_edit.text()
        self.config["orientation"] = self.side_combo.currentText()
        self.config["threads"] = self.threads_spin.value()
        self.config["hash_mb"] = self.hash_spin.value()
        self.config["multipv"] = self.multipv_spin.value()
        self.config["depth"] = self.depth_spin.value()
        self.config["movetime_ms"] = self.movetime_spin.value()
        self.config["skill_level"] = self.skill_spin.value()
        self.config["click_delay_ms"] = self.delay_spin.value()
        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.log_signal.emit(f"config save failed: {e}")

        self.log_signal.emit("starting engine...")
        self.start_btn.setEnabled(False)