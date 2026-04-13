"""独立 PPO 训练脚本 — 在运行 exp1 的 M3-PPO 之前手动执行一次.

Usage:
    python -m experiments.train_ppo
"""

import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.runner import load_config, per_method_dir
from src.baselines.ppo_policy import PPOTrainer


MODEL_DIR = "results/models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_model")
FIG_PATH = "results/figures/ppo_training_curve.png"
LEGACY_CURVE_CSV = "results/data/ppo_training_curve.csv"


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

    method_dir = per_method_dir("M3-PPO")
    curve_csv = str(method_dir / "training_curve.csv")

    print("=" * 60)
    print("PPO Training (mixed steady+burst scenario)")
    print("=" * 60)

    trainer = PPOTrainer(config)
    trainer.train(
        model_path=MODEL_PATH,
        curve_path=curve_csv,
        seed=config.get("ppo", {}).get("seed", 42),
        verbose=1,
    )

    # Copy to legacy location for the existing visualization script.
    shutil.copyfile(curve_csv, LEGACY_CURVE_CSV)

    plot_training_curve(curve_csv, FIG_PATH)

    print(f"\nModel:        {MODEL_PATH}.zip")
    print(f"Curve (main): {curve_csv}")
    print(f"Curve (copy): {LEGACY_CURVE_CSV}")
    print(f"Figure:       {FIG_PATH}")
    print("PPO training complete.\n")


if __name__ == "__main__":
    train_ppo()
