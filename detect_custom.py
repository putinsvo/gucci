#!/usr/bin/env python3
"""
detect_custom.py — распознавание позиции по скриншоту 2D-доски: YOLO -> сетка 8x8 -> FEN -> python-chess.

Запуск:
    python detect_custom.py board.png --weights runs/chess/train/weights/best.pt
"""
import argparse
import json
import re
import sys

import chess
import cv2
import numpy as np

SIZE = 640

# Имена классов: white_pawn, black_knight и т.п. Совпадение ТОЧНОЕ по всему слову,
# поэтому knight (-> n) никогда не путается с king (-> k).
CLASS_RE = re.compile(r"^(white|black)[\s_\-]*(pawn|knight|bishop|rook|queen|king)$")
PIECE_SYMBOL = {"pawn": "p", "knight": "n", "bishop": "b", "rook": "r", "queen": "q", "king": "k"}


def class_to_symbol(name):
    """'white_pawn' -> 'P', 'black_knight' -> 'n', 'black_king' -> 'k'. Прочее -> None."""
    m = CLASS_RE.match(name.strip().lower())
    if not m:
        return None
    color, piece = m.groups()
    sym = PIECE_SYMBOL[piece]
    return sym.upper() if color == "white" else sym


# ----------------------------------------------------------------------------
# Поиск и выравнивание доски
# ----------------------------------------------------------------------------
def order_points(pts):
    """Порядок: верх-лево, верх-право, низ-право, низ-лево."""
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def find_board(img, min_area_frac=0.2):
    """Canny + approxPolyDP: ищем выпуклый четырёхугольник, примерно квадратный и достаточно большой."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) < min_area_frac * img_area:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        _, _, bw, bh = cv2.boundingRect(approx)
        if not 0.75 <= bw / bh <= 1.33:       # плоская доска сверху — почти квадрат
            continue
        return order_points(approx.reshape(4, 2).astype("float32"))
    return None


def near_full_frame(corners, w, h, tol=0.03):
    """Доска и так занимает весь кадр -> warp не нужен (он только срежет пару пикселей)."""
    full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    return bool(np.all(np.abs(corners - full) <= tol * np.array([w, h])))


def warp(img, corners, size=SIZE, margin=0.0):
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(corners, dst), (size, size))
    if margin > 0:                             # срезать рамку доски, если она попала в кадр
        m = int(size * margin)
        out = cv2.resize(out[m:size - m, m:size - m], (size, size))
    return out


def align_board(img, margin=0.0):
    """Возвращает доску 640x640. Если контур не найден — берём всё изображение."""
    h, w = img.shape[:2]
    corners = find_board(img)
    if corners is None:
        print("Контур доски не найден — беру всё изображение.", file=sys.stderr)
        return cv2.resize(img, (SIZE, SIZE))
    if near_full_frame(corners, w, h):
        return cv2.resize(img, (SIZE, SIZE))
    return warp(img, corners, SIZE, margin)


# ----------------------------------------------------------------------------
# Детекции -> сетка 8x8 -> FEN
# ----------------------------------------------------------------------------
def build_grid(result, size=SIZE):
    """Сетка 8x8: клетка -> (символ FEN, conf, имя класса) или None.
    Клетка определяется по центру bbox; при конфликте побеждает бокс с большей уверенностью."""
    names, cell = result.names, size / 8
    grid = [[None] * 8 for _ in range(8)]
    boxes = result.boxes
    for (x1, y1, x2, y2), cls, conf in zip(boxes.xyxy.cpu().numpy(),
                                           boxes.cls.cpu().numpy().astype(int),
                                           boxes.conf.cpu().numpy()):
        sym = class_to_symbol(names[cls])
        if sym is None:
            print(f"Пропускаю неизвестный класс: {names[cls]}", file=sys.stderr)
            continue
        col = int(np.clip(((x1 + x2) / 2) // cell, 0, 7))
        row = int(np.clip(((y1 + y2) / 2) // cell, 0, 7))
        if grid[row][col] is None or conf > grid[row][col][1]:
            grid[row][col] = (sym, float(conf), names[cls])
    return grid


def grid_to_fen(grid, turn="w", castling="-"):
    rows = []
    for r in grid:
        out, empty = "", 0
        for cell in r:
            if cell is None:
                empty += 1
            else:
                out += (str(empty) if empty else "") + cell[0]
                empty = 0
        rows.append(out + (str(empty) if empty else ""))
    return "/".join(rows) + f" {turn} {castling} - 0 1"


def validate(fen):
    """(ok, список проблем) через python-chess."""
    try:
        board = chess.Board(fen)
    except ValueError as e:
        return False, [f"некорректный FEN: {e}"]
    status = board.status()
    if status == chess.Status.VALID:
        return True, []
    return False, [s.name for s in chess.Status if s != chess.Status.VALID and s in status]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="board.png")
    ap.add_argument("--weights", default="runs/chess/train/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.0, help="доля, срезаемая с краёв после warp (рамка)")
    ap.add_argument("--flip", action="store_true", help="белые сверху (доска повёрнута на 180°)")
    ap.add_argument("--turn", choices=["w", "b"], default="w")
    ap.add_argument("--castling", default="-", help="права на рокировку для FEN, например KQkq")
    ap.add_argument("--debug", default="debug_warped.png", help="файл с боксами ('' — не сохранять)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Не удалось прочитать {args.image}")

    board_img = align_board(img, args.margin)

    from ultralytics import YOLO               # импорт здесь: функции выше можно тестировать без ultralytics
    model = YOLO(args.weights)
    result = model.predict(board_img, imgsz=SIZE, conf=args.conf, device="cpu", verbose=False)[0]

    if len(result.boxes) == 0:
        print("Детекций нет: попробуйте --conf 0.1 и посмотрите debug_warped.png", file=sys.stderr)

    grid = build_grid(result)
    if args.flip:
        grid = [row[::-1] for row in grid[::-1]]

    fen = grid_to_fen(grid, args.turn, args.castling)
    ok, problems = validate(fen)

    if args.debug:                              # доска + боксы + сетка 8x8 (без GUI, только файл)
        vis = result.plot() if len(result.boxes) else board_img.copy()
        for i in range(1, 8):
            p = SIZE * i // 8
            cv2.line(vis, (p, 0), (p, SIZE), (0, 255, 0), 1)
            cv2.line(vis, (0, p), (SIZE, p), (0, 255, 0), 1)
        cv2.imwrite(args.debug, vis)

    matrix = [[c[2] if c else "empty" for c in row] for row in grid]
    if args.json:
        print(json.dumps({"fen": fen, "valid": ok, "problems": problems, "board": matrix},
                         ensure_ascii=False, indent=2))
    else:
        print("FEN:", fen)
        for row in matrix:
            print(" ".join(f"{n:>13}" for n in row))
        print("python-chess:", "OK, позиция допустима" if ok else f"НЕ прошла проверку: {', '.join(problems)}")


if __name__ == "__main__":
    main()