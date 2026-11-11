#!/usr/bin/env python3
"""Распознавание шахматной позиции по скриншоту 2D-доски: YOLOv8 -> FEN -> python-chess."""
import argparse
import json
import sys

import chess
import cv2
import numpy as np
from ultralytics import YOLO

SIZE = 640
TYPES = {"pawn": "p", "knight": "n", "horse": "n", "bishop": "b",
         "rook": "r", "queen": "q", "king": "k"}


def class_to_symbol(name: str):
    """'white_knight' / 'white-knight' / 'WhiteHorse' -> 'N'. Не шахматный класс -> None."""
    n = name.lower()
    color = "white" if "white" in n else "black" if "black" in n else None
    if color is None:
        return None
    for key, sym in TYPES.items():
        if key in n:
            return sym.upper() if color == "white" else sym
    return None


def order_points(pts):
    """Порядок: верх-лево, верх-право, низ-право, низ-лево."""
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def find_board(img, min_area_frac=0.2):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) < min_area_frac * img_area:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_points(approx.reshape(4, 2).astype("float32"))
    return None


def warp(img, corners, size=SIZE, margin=0.0):
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(corners, dst), (size, size))
    if margin > 0:  # срезать рамку доски, если она попала в кадр
        m = int(size * margin)
        out = cv2.resize(out[m:size - m, m:size - m], (size, size))
    return out


def build_grid(result, size, anchor_y):
    """Сетка 8x8: каждая клетка -> (символ FEN, conf, имя класса) или None."""
    names, cell = result.names, size / 8
    grid = [[None] * 8 for _ in range(8)]
    boxes = result.boxes
    for (x1, y1, x2, y2), cls, conf in zip(boxes.xyxy.cpu().numpy(),
                                           boxes.cls.cpu().numpy().astype(int),
                                           boxes.conf.cpu().numpy()):
        sym = class_to_symbol(names[cls])
        if sym is None:
            continue
        cx = (x1 + x2) / 2
        cy = y1 + (y2 - y1) * anchor_y
        col = int(np.clip(cx // cell, 0, 7))
        row = int(np.clip(cy // cell, 0, 7))
        if grid[row][col] is None or conf > grid[row][col][1]:
            grid[row][col] = (sym, float(conf), names[cls])
    return grid


def grid_to_fen(grid, turn="w"):
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
    return "/".join(rows) + f" {turn} - - 0 1"


def validate(fen):
    try:
        board = chess.Board(fen)
    except ValueError as e:
        return False, [f"malformed FEN: {e}"]
    st = board.status()
    if st == chess.Status.VALID:
        return True, []
    return False, [s.name for s in chess.Status if s != chess.Status.VALID and s in st]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="board.png")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--no-warp", action="store_true", help="картинка уже ровно обрезана")
    ap.add_argument("--margin", type=float, default=0.0, help="доля, срезаемая с краёв после warp (рамка)")
    ap.add_argument("--flip", action="store_true", help="белые сверху (доска повёрнута на 180°)")
    ap.add_argument("--turn", choices=["w", "b"], default="w")
    ap.add_argument("--anchor-y", type=float, default=0.5,
                    help="какая часть высоты bbox считается центром фигуры (0.5 = центр)")
    ap.add_argument("--debug", default="debug_warped.png", help="куда сохранить картинку с боксами ('' = не сохранять)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Не удалось прочитать {args.image}")

    if args.no_warp:
        board_img = cv2.resize(img, (SIZE, SIZE))
    else:
        corners = find_board(img)
        if corners is None:
            print("Контур доски не найден, использую всё изображение целиком.", file=sys.stderr)
            h, w = img.shape[:2]
            corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
        board_img = warp(img, corners, SIZE, args.margin)

    model = YOLO(args.weights)
    result = model.predict(board_img, imgsz=SIZE, conf=args.conf, verbose=False)[0]

    grid = build_grid(result, SIZE, args.anchor_y)
    if args.flip:
        grid = [row[::-1] for row in grid[::-1]]

    fen = grid_to_fen(grid, args.turn)
    ok, problems = validate(fen)

    if args.debug:
        vis = result.plot()
        for i in range(1, 8):
            p = int(SIZE * i / 8)
            cv2.line(vis, (p, 0), (p, SIZE), (0, 255, 0), 1)
            cv2.line(vis, (0, p), (SIZE, p), (0, 255, 0), 1)
        cv2.imwrite(args.debug, vis)

    names_matrix = [[c[2] if c else "empty" for c in row] for row in grid]
    if args.json:
        print(json.dumps({"fen": fen, "valid": ok, "problems": problems, "board": names_matrix},
                         ensure_ascii=False, indent=2))
    else:
        print("FEN:", fen)
        for row in names_matrix:
            print(" ".join(f"{n:>13}" for n in row))
        print("python-chess:", "OK, позиция допустима" if ok else f"НЕ прошла проверку: {', '.join(problems)}")


if __name__ == "__main__":
    main()