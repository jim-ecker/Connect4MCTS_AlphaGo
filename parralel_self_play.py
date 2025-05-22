# parallel_self_play.py

import os
import torch
import multiprocessing as mp
from connect4 import Connect4
from model import Connect4Net
from mcts import MCTS
from self_play import run_self_play, SelfPlayExample


def _worker_self_play(args):
    """
    Worker function for a single self-play game.
    args: (model_state_dict, num_simulations, seed)
    Returns a list of SelfPlayExample for one game.
    """
    model_state_dict, num_simulations, seed = args
    # Reconstruct model and MCTS for this process
    model = Connect4Net()
    model.load_state_dict(model_state_dict)
    mcts = MCTS(Connect4, model, num_simulations=num_simulations)
    # Optionally set a seed for reproducibility
    try:
        import numpy as _np
        _np.random.seed(seed)
        torch.manual_seed(seed)
    except Exception:
        pass
    # Run one self-play game
    examples = run_self_play(mcts, Connect4, num_games=1)
    return examples


def run_self_play_parallel(model, num_simulations=50, num_games=20, num_workers=None):
    """
    Parallelized self-play: runs num_games games across multiple processes.

    Args:
        model (Connect4Net): the neural network guiding MCTS.
        num_simulations (int): MCTS simulations per move.
        num_games (int): total self-play games to generate.
        num_workers (int): number of parallel processes (defaults to CPU count).

    Returns:
        List[SelfPlayExample]: flattened list of all generated examples.
    """
    # Serialize model once
    model_state_dict = model.state_dict()
    # Determine worker count
    if num_workers is None:
        num_workers = min(mp.cpu_count(), num_games)

    # Prepare arguments per game
    args = [(model_state_dict, num_simulations, i) for i in range(num_games)]

    # Run in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_self_play, args)

    # Flatten list of lists
    data = []
    for game_examples in results:
        data.extend(game_examples)
    return data

