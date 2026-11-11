import time
from typing import Callable

import pydirectinput


class Mover:
    def __init__(self, click_delay_ms: int = 80, halt_flag: Callable[[], bool] = lambda: False):
        self.click_delay = click_delay_ms / 1000.0
        self._halt = halt_flag
        pydirectinput.PAUSE = 0
        pydirectinput.FAILSAFE = True

    def _click(self, x: int, y: int):
        if self._halt():
            raise RuntimeError("halted")
        pydirectinput.moveTo(x, y)
        time.sleep(0.03)
        pydirectinput.mouseDown()
        time.sleep(0.02)
        pydirectinput.mouseUp()
        time.sleep(self.click_delay)

    def make_move(self, from_xy: tuple[int, int], to_xy: tuple[int, int]):
        try:
            self._click(*from_xy)
            self._click(*to_xy)
        finally:
            # защита от зажатой кнопки
            pydirectinput.mouseUp()