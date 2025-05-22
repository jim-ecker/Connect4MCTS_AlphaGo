# self_play.py

from collections import namedtuple
import numpy as np
from connect4 import Connect4
from mcts import MCTS

SelfPlayExample = namedtuple('SelfPlayExample', ['state', 'pi', 'z'])

def run_self_play(mcts, game_cls, num_games=1):
    data = []

    for _ in range(num_games):
        game = game_cls()
        trajectory = []
        while not game.is_terminal():
            state_tensor = game.get_state_tensor()
            pi = mcts.run(game)
            action = np.random.choice(len(pi), p=pi)
            trajectory.append((state_tensor, pi, game.current_player))
            game.make_move(action)

        winner = game.get_winner()
        for state, pi, player in trajectory:
            z = 0 if winner is None else 1 if winner == player else -1
            data.append(SelfPlayExample(state, pi, z))

    return data

