"""一键运行所有实验并生成图表 (顺序执行版本).

推荐并行工作流 (更快):
    python -m experiments.train_ppo                      # 先训练 PPO
    python -m experiments.exp1_performance M1-Fixed      # 终端1
    python -m experiments.exp1_performance M2-Threshold  # 终端2
    python -m experiments.exp1_performance M3-PPO        # 终端3
    python -m experiments.exp1_performance M4-SingleLLM  # 终端4
    python -m experiments.exp1_performance M5-PA         # 终端5
    python -m experiments.exp1_performance M5-PC         # 终端6
    python -m experiments.exp2_negotiation M5-NoNeg      # 终端7
    # 全部完成后:
    python -m experiments.aggregate_results
    python -m visualization.plot_results

本脚本把上述步骤串行执行一遍, 主要用于本机完整跑一遍流程.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from experiments.aggregate_results import aggregate_all
from experiments.exp1_performance import EXP1_METHODS, run_exp1_method
from experiments.exp2_negotiation import run_exp2_method
from experiments.runner import load_config
from experiments.train_ppo import train_ppo
from visualization.plot_results import generate_all_plots


PPO_MODEL_PATH = "results/models/ppo_model.zip"


def main():
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    config = load_config()
    t_start = time.time()

    # 1) 训练 PPO (若模型不存在)
    if not os.path.exists(PPO_MODEL_PATH):
        print("=" * 60)
        print("Step 1: Training PPO (model not found)")
        print("=" * 60)
        train_ppo(config)
    else:
        print(f"[SKIP] PPO model already exists at {PPO_MODEL_PATH}")

    # 2) 实验1: 逐 method 顺序运行
    print("\n" + "=" * 60)
    print("Step 2: 实验1 (M1..M5) — 逐 method 顺序运行")
    print("=" * 60)
    for method in EXP1_METHODS:
        run_exp1_method(method, config)

    # 3) 实验2: 仅需补跑 M5-NoNeg (M5-PA/M5-PC 已由实验1产出)
    print("\n" + "=" * 60)
    print("Step 3: 实验2 — 补跑 M5-NoNeg")
    print("=" * 60)
    run_exp2_method("M5-NoNeg", config)

    # 4) 聚合 per-method 结果
    print("\n" + "=" * 60)
    print("Step 4: 聚合 per-method 结果")
    print("=" * 60)
    aggregate_all()

    # 5) 生成可视化
    print("\n" + "=" * 60)
    print("Step 5: 生成可视化图表")
    print("=" * 60)
    generate_all_plots()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"全部实验完成! 总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"数据文件: results/data/")
    print(f"图表文件: results/figures/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
