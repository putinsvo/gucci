import threading
import keyboard


class HotkeyManager:
    def __init__(self, toggle_key: str = "f8", stop_key: str = "f12"):
        self.toggle_key = toggle_key
        self.stop_key = stop_key
        self.stop_event = threading.Event()
        self.toggle_event = threading.Event()
        self._handles = []

    def register(self):
        self._handles.append(keyboard.add_hotkey(self.toggle_key, self._toggle, suppress=False))
        self._handles.append(keyboard.add_hotkey(self.stop_key, self._emergency_stop, suppress=False))

    def _toggle(self):
        self.toggle_event.set()

    def _emergency_stop(self):
        self.stop_event.set()

    def clear_stop(self):
        self.stop_event.clear()

    def unregister(self):
        for h in self._handles:
            try:
                keyboard.remove_hotkey(h)
            except Exception:
                pass
        self._handles.clear()