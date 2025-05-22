import numpy as np

ROWS, COLS = 6, 7

class Connect4:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = 1
        self.last_move = None

    def clone(self):
        clone = Connect4()
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.last_move = self.last_move
        return clone

    def get_legal_actions(self):
        return [c for c in range(COLS) if self.board[0, c] == 0]

    def make_move(self, col):
        for row in reversed(range(ROWS)):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.last_move = (row, col)
                self.current_player *= -1
                return True
        return False

    def is_terminal(self):
        return self.get_winner() is not None or len(self.get_legal_actions()) == 0

    def get_winner(self):
        if self.last_move is None:
            return None

        r, c = self.last_move
        player = self.board[r, c]

        def count_dir(dr, dc):
            count = 0
            for i in range(1, 4):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < ROWS and 0 <= nc < COLS and self.board[nr, nc] == player:
                    count += 1
                else:
                    break
            return count

        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1 + count_dir(dr, dc) + count_dir(-dr, -dc)
            if count >= 4:
                return player
        return None

    def get_state_tensor(self):
        p1 = (self.board == self.current_player).astype(np.float32)
        p2 = (self.board == -self.current_player).astype(np.float32)
        return np.stack([p1, p2])

