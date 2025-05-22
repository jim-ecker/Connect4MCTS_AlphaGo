# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from self_play import run_self_play, SelfPlayExample
from model import Connect4Net
import numpy as np
import os

class Connect4Dataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, pi, z = self.examples[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(pi, dtype=torch.float32),
            torch.tensor([z], dtype=torch.float32)
        )

def train_model(model, dataset, batch_size=32, epochs=5, lr=1e-3, checkpoint_path=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn_value = nn.MSELoss()
    loss_fn_policy = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for state, pi, z in dataloader:
            optimizer.zero_grad()
            logits, value = model(state)
            loss_p = loss_fn_policy(logits, torch.argmax(pi, dim=1))
            loss_v = loss_fn_value(value.view(-1), z.view(-1))
            loss = loss_p + loss_v
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        if checkpoint_path:
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_path, f"connect4_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Model checkpoint saved to: {checkpoint_file}")

            if elo > best_elo:
                best_elo = elo
                torch.save(model.state_dict(), "checkpoints/best_model.pt")
                print("New best model saved as best_model.pt")
            print(f"Model checkpoint saved to: {checkpoint_file}")

if __name__ == "__main__":
    from connect4 import Connect4
    from mcts import MCTS
    from evaluate import evaluate_against_random

    def update_elo(current_elo, expected, actual, k=32):
        return current_elo + k * (actual - expected)

    num_iterations = 5
    games_per_iteration = 10
    elo = 1000
    opponent_elo = 1000

    best_elo = elo

    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1}/{num_iterations} ===")
        model = Connect4Net()
        latest_checkpoint = f"checkpoints/connect4_epoch{i}.pt" if i > 0 else None
        if latest_checkpoint and os.path.exists(latest_checkpoint):
            model.load_state_dict(torch.load(latest_checkpoint))

        mcts = MCTS(Connect4, model, num_simulations=50)
        examples = run_self_play(mcts, Connect4, num_games=games_per_iteration)
        dataset = Connect4Dataset(examples)
        train_model(model, dataset, checkpoint_path="checkpoints")

        checkpoint_file = f"checkpoints/connect4_epoch{i+1}.pt"
        if os.path.exists(checkpoint_file):
            result = evaluate_against_random(checkpoint_file, num_games=20)

            total_games = result['games']
            score = result['wins'] + 0.5 * result['draws']
            expected_score = 1 / (1 + 10 ** ((opponent_elo - elo) / 400))
            elo = update_elo(elo, expected_score * total_games, score)

            with open("win_rates.csv", "a") as f:
                f.write(f"{i+1},{result['wins']},{result['losses']},{result['draws']},{int(elo)}\n")

