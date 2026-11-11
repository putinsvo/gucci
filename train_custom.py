#!/usr/bin/env python3
"""
train_custom.py — генерация YOLO-датасета из ОДНОГО скриншота стартовой позиции
и обучение yolov8n на CPU.

Что делает:
  1. Режет board.png на 8x8 клеток, размечает фигуры по стартовому FEN
     (бокс в центре клетки, с отступом от краёв клетки, чтобы не задеть сетку).
  2. Делает 3 аугментации исходного скрина (сдвиг / масштаб / яркость).
  3. Чтобы модель не выучила одну-единственную расстановку, собирает
     синтетические доски: вырезанные из скрина фигуры случайно переставляются
     по клеткам. Разметка при этом точная.
  4. Пишет datasets/chess/{images,labels}/{train,val} и data.yaml (12 классов).
  5. Запускает обучение YOLOv8n на CPU и печатает путь к best.pt.

Запуск:
    python train_custom.py                       # всё сразу
    python train_custom.py --no-train            # только сгенерировать датасет
    python train_custom.py --epochs 30 --synthetic 40   # быстрее
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

SIZE = 640                 # размер доски в пикселях (скрин 640x640)
CELL = SIZE // 8           # 80 px на клетку
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

# Порядок классов ОБЯЗАН совпадать с SYM2ID ниже и с data.yaml
CLASSES = [
    "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king",
]
SYM2ID = {s: i for i, s in enumerate("PNBRQKpnbrqk")}


# ----------------------------------------------------------------------------
# Разбор FEN и нарезка доски
# ----------------------------------------------------------------------------
def fen_to_squares(fen):
    """FEN -> {(row, col): символ}. row=0 — верхний ряд (8-я горизонталь), col=0 — вертикаль a."""
    squares = {}
    for r, row in enumerate(fen.split()[0].split("/")):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            else:
                squares[(r, c)] = ch
                c += 1
    return squares


def load_board(path):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Не удалось прочитать {path}")
    if img.shape[:2] != (SIZE, SIZE):
        print(f"[!] {path} имеет размер {img.shape[1]}x{img.shape[0]}, приводим к {SIZE}x{SIZE}")
        img = cv2.resize(img, (SIZE, SIZE))
    return img


def cut_cells(board):
    """Нарезка на 64 клетки. Клетка (0,0) — a8, светлая, поэтому светлые клетки: (r+c) % 2 == 0."""
    return {(r, c): board[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL].copy()
            for r in range(8) for c in range(8)}


def empty_templates(cells):
    """Эталон пустой клетки каждого цвета: медиана по пустым клеткам 3–6 горизонталей.
    Медиана заодно убирает случайные мелочи (подписи координат и т.п.)."""
    tpl = {}
    for parity in (0, 1):
        stack = [cells[(r, c)] for r in range(2, 6) for c in range(8) if (r + c) % 2 == parity]
        tpl[parity] = np.median(np.stack(stack), axis=0).astype(np.uint8)
    return tpl


def build_piece_pool(cells, fen):
    """Для каждого символа — список (цвет клетки, картинка клетки с фигурой)."""
    pool = {}
    for (r, c), sym in fen_to_squares(fen).items():
        pool.setdefault(sym, []).append(((r + c) % 2, cells[(r, c)]))
    return pool


# ----------------------------------------------------------------------------
# Синтетические доски: перестановка фигур
# ----------------------------------------------------------------------------
def piece_mask(cell, empty):
    """Мягкая маска фигуры = отличие клетки от пустой клетки того же цвета.
    Контуры заливаются, чтобы «пустое» нутро светлых фигур не выпало из маски."""
    diff = cv2.absdiff(cell, empty).max(axis=2)
    binary = (diff > 30).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    filled = cv2.dilate(filled, np.ones((3, 3), np.uint8))          # захватить сглаженный край
    soft = cv2.GaussianBlur(filled, (3, 3), 0).astype(np.float32) / 255.0
    return soft[..., None]


def piece_on(sym, parity, pool, empties):
    """Картинка фигуры на клетке нужного цвета. Если в стартовой позиции такой
    комбинации нет (напр. ферзь есть только на одном цвете), переносим фигуру по маске."""
    same = [cell for p, cell in pool[sym] if p == parity]
    if same:
        return random.choice(same)
    p, cell = random.choice(pool[sym])
    m = piece_mask(cell, empties[p])
    out = empties[parity].astype(np.float32) * (1 - m) + cell.astype(np.float32) * m
    return out.astype(np.uint8)


def random_position():
    """Случайная расстановка: по королю на сторону, остальных — случайное число
    (в пределах комплекта), пешки не на 1-й и 8-й горизонталях."""
    free = [(r, c) for r in range(8) for c in range(8)]
    random.shuffle(free)
    pieces = []
    for upper in (True, False):
        counts = {"p": random.randint(0, 8), "n": random.randint(0, 2), "b": random.randint(0, 2),
                  "r": random.randint(0, 2), "q": random.randint(0, 1), "k": 1}
        for t, n in counts.items():
            pieces += [t.upper() if upper else t] * n
    random.shuffle(pieces)
    placement = {}
    for sym in pieces:
        for i, (r, c) in enumerate(free):
            if sym in "Pp" and r in (0, 7):
                continue
            placement[(r, c)] = sym
            free.pop(i)
            break
    return placement


def compose(placement, empties, pool):
    """Собираем доску из целых клеток (стыкуются без швов)."""
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    for r in range(8):
        for c in range(8):
            parity = (r + c) % 2
            sym = placement.get((r, c))
            tile = piece_on(sym, parity, pool, empties) if sym else empties[parity]
            img[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = tile
    return img


# ----------------------------------------------------------------------------
# Разметка и аугментации
# ----------------------------------------------------------------------------
def make_labels(placement, inset):
    """Бокс в центре клетки; inset — доля клетки, отступаемая с каждой стороны (0.10 -> 64 из 80 px)."""
    side = CELL * (1 - 2 * inset)
    return [(SYM2ID[sym], (c + 0.5) * CELL, (r + 0.5) * CELL, side, side)
            for (r, c), sym in placement.items()]


def augment(img, labels, dx=0.0, dy=0.0, scale=1.0, alpha=1.0, beta=0.0):
    """Сдвиг + масштаб относительно центра + яркость/контраст. Боксы пересчитываются."""
    h, w = img.shape[:2]
    tx = (1 - scale) * (w / 2) + dx
    ty = (1 - scale) * (h / 2) + dy
    m = np.float32([[scale, 0, tx], [0, scale, ty]])
    out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    new = []
    for cls, bx, by, bw, bh in labels:
        x1, x2 = scale * (bx - bw / 2) + tx, scale * (bx + bw / 2) + tx
        y1, y2 = scale * (by - bh / 2) + ty, scale * (by + bh / 2) + ty
        cx1, cx2, cy1, cy2 = max(x1, 0), min(x2, w), max(y1, 0), min(y2, h)
        # бокс, ушедший за край больше чем наполовину, выбрасываем
        if (cx2 - cx1) < 0.5 * (x2 - x1) or (cy2 - cy1) < 0.5 * (y2 - y1):
            continue
        new.append((cls, (cx1 + cx2) / 2, (cy1 + cy2) / 2, cx2 - cx1, cy2 - cy1))
    return out, new


def save_sample(root, split, name, img, labels):
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "images" / split / f"{name}.png"), img)
    lines = [f"{cls} {cx / SIZE:.6f} {cy / SIZE:.6f} {w / SIZE:.6f} {h / SIZE:.6f}"
             for cls, cx, cy, w, h in labels]
    (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")


def random_aug(img, labels):
    return augment(img, labels,
                   dx=random.uniform(-10, 10), dy=random.uniform(-10, 10),
                   scale=random.uniform(0.97, 1.03),
                   alpha=random.uniform(0.9, 1.1), beta=random.uniform(-8, 8))


# ----------------------------------------------------------------------------
# Датасет
# ----------------------------------------------------------------------------
FIXED_AUGS = [  # 3 детерминированные аугментации исходного скрина
    dict(dx=8, dy=-6, scale=1.00, alpha=1.10, beta=5),
    dict(dx=-10, dy=7, scale=0.97, alpha=0.92, beta=-5),
    dict(dx=4, dy=10, scale=1.03, alpha=1.00, beta=0),
]


def build_dataset(board_path, root, n_synth, n_val_synth, inset):
    root = Path(root)
    for sub in ("images", "labels"):
        shutil.rmtree(root / sub, ignore_errors=True)       # чистая пересборка

    board = load_board(board_path)
    placement = fen_to_squares(START_FEN)
    labels = make_labels(placement, inset)

    # исходный скрин + 3 аугментации
    save_sample(root, "train", "orig", board, labels)
    for i, params in enumerate(FIXED_AUGS):
        img, lab = augment(board, labels, **params)
        save_sample(root, "train", f"aug{i}", img, lab)

    # синтетические перестановки
    cells = cut_cells(board)
    empties = empty_templates(cells)
    pool = build_piece_pool(cells, START_FEN)

    def synth(n, split):
        for i in range(n):
            pos = random_position()
            img = compose(pos, empties, pool)
            lab = make_labels(pos, inset)
            if random.random() < 0.7:
                img, lab = random_aug(img, lab)
            save_sample(root, split, f"synth{i:03d}", img, lab)

    synth(n_synth, "train")
    synth(n_val_synth, "val")
    save_sample(root, "val", "orig", board, labels)          # реальный скрин — тоже в val

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    n_train = 1 + len(FIXED_AUGS) + n_synth
    print(f"Датасет готов: {root}  (train: {n_train}, val: {n_val_synth + 1})")
    return yaml_path


# ----------------------------------------------------------------------------
# Обучение
# ----------------------------------------------------------------------------
def train(yaml_path, epochs, imgsz):
    from ultralytics import YOLO          # импорт здесь, чтобы --no-train работал без ultralytics

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path), epochs=epochs, imgsz=imgsz, device="cpu",
        batch=8, workers=2,
        project=str(Path("runs/chess").resolve()), name="train", exist_ok=True,
        fliplr=0.0,      # зеркалить нельзя: фигуры смотрят в фиксированную сторону
        hsv_h=0.0,       # не менять оттенок: цвета доски и фигур — часть признака
        mosaic=0.5,
        plots=False,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nГотово. Лучшие веса: {best.resolve()}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="board.png")
    ap.add_argument("--out", default="datasets/chess")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--synthetic", type=int, default=60, help="число синтетических досок в train")
    ap.add_argument("--val-synthetic", type=int, default=10)
    ap.add_argument("--inset", type=float, default=0.10, help="отступ бокса от края клетки (доля клетки)")
    ap.add_argument("--no-train", action="store_true", help="только сгенерировать датасет")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    yaml_path = build_dataset(args.board, args.out, args.synthetic, args.val_synthetic, args.inset)
    if not args.no_train:
        train(yaml_path, args.epochs, args.imgsz)


if __name__ == "__main__":
    main()