"""独立PPO训练脚本 — 在运行exp1之前手动执行一次。

Usage:
    python -m experiments.train_ppo
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.runner import load_config
from src.baselines.ppo_policy import PPOTrainer


MODEL_DIR = "results/models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_model")
CURVE_CSV = "results/data/ppo_training_curve.csv"
CURVE_FIG = "results/figures/ppo_training_curve.png"


def plot_training_curve(csv_path: str, fig_path: str):
    """Generate the PPO training convergence plot."""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["episode"], df["reward"], alpha=0.3, color="#3498db",
            linewidth=0.8, label="Episode reward")

    window = min(50, len(df))
    smoothed = df["reward"].rolling(window, min_periods=1).mean()
    ax.plot(df["episode"], smoothed, color="#e74c3c", linewidth=2,
            label=f"Rolling mean (w={window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("PPO Training Convergence Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curve figure saved to {fig_path}")


def train_ppo(config: dict | None = None):
    config = config or load_config()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 60)
    print("PPO Training (mixed steady+burst scenario)")
    print("=" * 60)

    trainer = PPOTrainer(config)
    trainer.train(model_path=MODEL_PATH, curve_path=CURVE_CSV, verbose=1)

    plot_training_curve(CURVE_CSV, CURVE_FIG)

    print(f"\nModel:  {MODEL_PATH}.zip")
    print(f"Curve:  {CURVE_CSV}")
    print(f"Figure: {CURVE_FIG}")
    print("PPO training complete.\n")


if __name__ == "__main__":
    train_ppo()
