import chess
import chess.engine
import cv2
from piece_recognize import PieceRecognizer
from fen import board_to_fen

# 1. Загружаем картинку
img = cv2.imread("board.png")
if img is None:
    print("Ошибка: board.png не найден в корне проекта!")
    exit()

# 2. Распознаем доску
rec = PieceRecognizer()
# Если шаблонов нет, они не загрузятся, и он ничего не распознает
if not rec.templates:
    print("Ошибка: шаблоны фигур не найдены! Запусти 'python tools/gen_templates.py'")
    exit()

board = rec.recognize_board(img, orientation="white")

print("Распознанная доска:")
for row in board:
    print("".join([p if p else "." for p in row]))

# 3. Собираем FEN
fen = board_to_fen(board, side_to_move="w")
print(f"\nFEN: {fen}")

b = chess.Board(fen)
if not b.is_valid():
    print("Ошибка: FEN невалидный (фигуры распознались криво).")
    exit()

# 4. Запускаем Stockfish
# В Ubuntu он установлен сюда:
engine_path = "/usr/games/stockfish" 
try:
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 2, "Hash": 128})

    print("\nАнализирую (глубина 15)...")
    result = engine.play(b, chess.engine.Limit(depth=15))
    print(f"Лучший ход от движка: {result.move}")
    engine.quit()
except Exception as e:
    print(f"Ошибка при запуске движка: {e}")