# plot_win_rates.py

import matplotlib.pyplot as plt
import pandas as pd

# Load win rate and Elo data
win_rates = pd.read_csv("win_rates.csv", names=["iteration", "wins", "losses", "draws", "elo"])

# Plot win/loss/draw
plt.figure(figsize=(10, 6))
plt.plot(win_rates["iteration"], win_rates["wins"], label="Wins", marker='o')
plt.plot(win_rates["iteration"], win_rates["losses"], label="Losses", marker='x')
plt.plot(win_rates["iteration"], win_rates["draws"], label="Draws", marker='s')
plt.xlabel("Training Iteration")
plt.ylabel("Games")
plt.title("Model Performance vs Random Agent Over Training Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("win_rate_plot.png")
plt.show()

# Plot Elo rating
plt.figure(figsize=(10, 6))
plt.plot(win_rates["iteration"], win_rates["elo"], label="Elo Rating", marker='d', color='purple')
plt.xlabel("Training Iteration")
plt.ylabel("Elo Rating")
plt.title("Model Elo Rating Over Training Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("elo_rating_plot.png")
plt.show()

