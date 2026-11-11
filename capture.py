import mss
import numpy as np
import pygetwindow as gw


class ScreenCapture:
    def __init__(self, window_title: str):
        self.window_title = window_title
        self.sct = mss.mss()

    def _window_bbox(self):
        wins = [w for w in gw.getWindowsWithTitle(self.window_title) if w.width > 100 and w.height > 100]
        if not wins:
            return None
        w = wins[0]
        return {"left": w.left, "top": w.top, "width": w.width, "height": w.height}

    def grab_window(self) -> np.ndarray | None:
        bbox = self._window_bbox()
        if bbox is None:
            return None
        img = np.array(self.sct.grab(bbox))
        return img[:, :, :3]  # BGRA -> BGR

    def window_origin(self) -> tuple[int, int]:
        bbox = self._window_bbox()
        if bbox is None:
            return (0, 0)
        return (bbox["left"], bbox["top"])