import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class EngineOptions:
    threads: int = 4
    hash_mb: int = 256
    multipv: int = 3
    skill_level: int | None = 20
    depth: int = 18
    movetime_ms: int = 1000
    contempt: int | None = None


@dataclass
class PVLine:
    multipv: int
    depth: int
    score_cp: int | None
    score_mate: int | None
    pv: list[str] = field(default_factory=list)

    @property
    def score_str(self) -> str:
        if self.score_mate is not None:
            return f"#{self.score_mate}"
        if self.score_cp is not None:
            return f"{self.score_cp / 100:+.2f}"
        return "?"


class UCIEngine:
    def __init__(self, exe_path: str):
        self.exe_path = exe_path
        self.proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None
        self.options = EngineOptions()

    def start(self):
        if not os.path.isfile(self.exe_path):
            raise FileNotFoundError(self.exe_path)
        self.proc = subprocess.Popen(
            [self.exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._send("uci")
        self._wait_for("uciok", timeout=5.0)
        self.apply_options()
        self._send("isready")
        self._wait_for("readyok", timeout=10.0)

    def _read_loop(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self._lines.put(line.rstrip("\n"))

    def _send(self, cmd: str):
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("engine not started")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, timeout: float = 10.0):
        end = time.time() + timeout
        while time.time() < end:
            try:
                line = self._lines.get(timeout=0.05)
            except queue.Empty:
                continue
            if line.startswith(token):
                return line
        raise TimeoutError(f"engine did not reply '{token}'")

    def apply_options(self):
        o = self.options
        self._send(f"setoption name Threads value {o.threads}")
        self._send(f"setoption name Hash value {o.hash_mb}")
        self._send(f"setoption name MultiPV value {o.multipv}")
        if o.skill_level is not None:
            self._send(f"setoption name Skill Level value {o.skill_level}")
        if o.contempt is not None:
            self._send(f"setoption name Contempt value {o.contempt}")

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def analyze(self, fen: str, depth: int | None = None, movetime_ms: int | None = None):
        depth = depth if depth is not None else self.options.depth
        movetime_ms = movetime_ms if movetime_ms is not None else self.options.movetime_ms

        self._send(f"position fen {fen}")
        if movetime_ms and movetime_ms > 0:
            self._send(f"go movetime {movetime_ms}")
        else:
            self._send(f"go depth {depth}")

        lines: dict[int, PVLine] = {}
        bestmove = None
        while True:
            line = self._lines.get()
            if line.startswith("info ") and " pv " in line:
                parsed = self._parse_info(line)
                if parsed:
                    lines[parsed.multipv] = parsed
            elif line.startswith("bestmove"):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else None
                break
        return sorted(lines.values(), key=lambda x: x.multipv), bestmove

    @staticmethod
    def _parse_info(line: str) -> PVLine | None:
        toks = line.split()
        d = {"multipv": 1, "depth": 0, "score_cp": None, "score_mate": None, "pv": []}
        i = 1
        while i < len(toks):
            t = toks[i]
            if t == "depth" and i + 1 < len(toks):
                d["depth"] = int(toks[i + 1]); i += 2
            elif t == "multipv" and i + 1 < len(toks):
                d["multipv"] = int(toks[i + 1]); i += 2
            elif t == "score" and i + 2 < len(toks):
                kind, val = toks[i + 1], toks[i + 2]
                if kind == "cp":
                    d["score_cp"] = int(val)
                elif kind == "mate":
                    d["score_mate"] = int(val)
                i += 3
            elif t == "pv":
                d["pv"] = toks[i + 1:]; break
            else:
                i += 1
        if not d["pv"]:
            return None
        return PVLine(**d)

    def stop(self):
        if self.proc:
            try:
                self._send("quit")
            except Exception:
                pass
            self.proc.terminate()
            self.proc = None