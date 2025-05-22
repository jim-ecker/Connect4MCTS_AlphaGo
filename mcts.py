# mcts.py

import math
import numpy as np
import random
from collections import defaultdict

class MCTSNode:
    def __init__(self, game_state, parent=None, prior=0.0):
        self.game_state = game_state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

class MCTS:
    def __init__(self, game_cls, neural_net, num_simulations=100, c_puct=1.0):
        self.game_cls = game_cls
        self.nn = neural_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def run(self, root_game):
        root = MCTSNode(root_game)
        policy, _ = self.nn.predict(root_game.get_state_tensor())
        valid_moves = root_game.get_legal_actions()
        policy = self._mask_invalid(policy, valid_moves)
        for a in valid_moves:
            new_game = root_game.clone()
            new_game.make_move(a)
            root.children[a] = MCTSNode(new_game, root, prior=policy[a])

        for _ in range(self.num_simulations):
            self._simulate(root)

        # Create final policy based on visit counts
        visit_counts = np.array([root.children[a].visit_count if a in root.children else 0 for a in range(7)])
        return visit_counts / np.sum(visit_counts)

    def _simulate(self, node):
        path = [node]
        while node.expanded():
            action, node = self._select(node)
            path.append(node)

        game = node.game_state
        if game.is_terminal():
            winner = game.get_winner()
            value = 0 if winner is None else (1 if winner == game.current_player else -1)
        else:
            policy, value = self.nn.predict(game.get_state_tensor())
            valid_moves = game.get_legal_actions()
            policy = self._mask_invalid(policy, valid_moves)
            for a in valid_moves:
                next_game = game.clone()
                next_game.make_move(a)
                node.children[a] = MCTSNode(next_game, node, prior=policy[a])

        self._backpropagate(path, value)

    def _select(self, node):
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = -float('inf')
        best_action = None
        best_child = None
        for action, child in node.children.items():
            ucb = child.value() + self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
        return best_action, best_child

    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value
            value = -value  # Switch perspective

    def _mask_invalid(self, policy, valid_moves):
        masked = np.zeros_like(policy)
        for a in valid_moves:
            masked[a] = policy[a]
        if np.sum(masked) == 0:
            return np.ones_like(policy) / len(policy)
        return masked / np.sum(masked)

