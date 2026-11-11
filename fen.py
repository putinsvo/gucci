import chess


def board_to_fen(board, side_to_move="w", castling="KQkq", ep="-"):
    rows = []
    for r in range(8):
        empty = 0
        out = ""
        for c in range(8):
            p = board[r][c]
            if p is None:
                empty += 1
            else:
                if empty:
                    out += str(empty); empty = 0
                out += p
        if empty:
            out += str(empty)
        rows.append(out)
    return "/".join(rows) + f" {side_to_move} {castling} {ep} 0 1"


def validate(fen: str) -> bool:
    try:
        return chess.Board(fen).is_valid()
    except Exception:
        return False


def side_to_move_from_diff(prev_board, curr_board) -> str | None:
    """Грубо: если фигура переместилась, ход был за той стороной, чья фигура исчезла/сдвинулась.
    Возвращает сторону, которой ходить следующим: 'w' или 'b', либо None если разницы нет."""
    if prev_board is None:
        return None
    diff_white = 0
    diff_black = 0
    for r in range(8):
        for c in range(8):
            if prev_board[r][c] != curr_board[r][c]:
                p = prev_board[r][c]
                if p is not None:
                    if p.isupper():
                        diff_white += 1
                    else:
                        diff_black += 1
    if diff_white == 0 and diff_black == 0:
        return None
    # кто сдвинулся, тот и ходил; следующий ход — за противоположной стороной
    return "b" if diff_white >= diff_black else "w"


def square_to_rc(sq: str, orientation: str = "white") -> tuple[int, int]:
    file = ord(sq[0]) - ord("a")
    rank = int(sq[1]) - 1
    if orientation == "white":
        return 7 - rank, file
    return rank, 7 - file