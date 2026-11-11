import chess
import chess.engine
import cv2
import sys
from detect_custom import align_board, build_grid, grid_to_fen
from ultralytics import YOLO

# 1. Загружаем картинку и выравниваем доску
img = cv2.imread("board.png")
board_img = align_board(img)

# 2. Загружаем нашу обученную модель
model = YOLO("runs/chess/train/weights/best.pt")
result = model.predict(board_img, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]

# 3. Собираем FEN
grid = build_grid(result)
fen = grid_to_fen(grid, turn="w")
print(f"Распознанный FEN: {fen}")

# 4. Отдаем в Stockfish
engine_path = "/usr/games/stockfish"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)
engine.configure({"Threads": 2, "Hash": 128})

board = chess.Board(fen)
print("Stockfish думает...")
result = engine.play(board, chess.engine.Limit(depth=15))
print(f"Лучший ход: {result.move}")

engine.quit()