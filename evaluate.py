# evaluate.py

import torch
import numpy as np
from connect4 import Connect4
from mcts import MCTS
from model import Connect4Net


def play_game(player1, player2):
    game = Connect4()
    players = {1: player1, -1: player2}

    while not game.is_terminal():
        current_player = game.current_player
        action = players[current_player](game)
        game.make_move(action)

    return game.get_winner()


def random_player(game):
    return np.random.choice(game.get_legal_actions())


def mcts_player(model, num_simulations=25):
    def player(game):
        mcts = MCTS(Connect4, model, num_simulations)
        policy = mcts.run(game)
        return np.argmax(policy)
    return player


def evaluate_against_random(model_path, num_games=20):
    model = Connect4Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        if i % 2 == 0:
            winner = play_game(mcts_player(model), random_player)
        else:
            winner = play_game(random_player, mcts_player(model))

        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1

    print(f"Eval vs Random ({num_games} games): Wins: {wins}, Losses: {losses}, Draws: {draws}")
    return {"wins": wins, "losses": losses, "draws": draws, "games": num_games}


def evaluate_model_vs_model(model_path_1, model_path_2, num_games=20):
    model1 = Connect4Net()
    model1.load_state_dict(torch.load(model_path_1))
    model1.eval()

    model2 = Connect4Net()
    model2.load_state_dict(torch.load(model_path_2))
    model2.eval()

    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        if i % 2 == 0:
            winner = play_game(mcts_player(model1), mcts_player(model2))
        else:
            winner = play_game(mcts_player(model2), mcts_player(model1))
            if winner is not None:
                winner *= -1  # invert since model1 played second

        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1

    print(f"Model1 vs Model2 ({num_games} games): Wins: {wins}, Losses: {losses}, Draws: {draws}")
    return {"wins": wins, "losses": losses, "draws": draws, "games": num_games}

