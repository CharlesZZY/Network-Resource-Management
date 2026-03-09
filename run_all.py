"""一键运行所有实验并生成图表"""

from visualization.plot_results import generate_all_plots
from experiments.runner import load_config
from experiments.exp2_negotiation import run_exp2
from experiments.exp1_performance import run_exp1
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


def main():
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    config = load_config()
    t_start = time.time()

    # 实验1
    print("=" * 60)
    print("实验1: 核心性能对比 (M1, M2, M4, M5-PA, M5-PC)")
    print("=" * 60)
    df_ep1, df_st1 = run_exp1(config)

    # 实验2 (复用实验1的M5-PA/M5-PC数据)
    print("\n" + "=" * 60)
    print("实验2: 协商机制验证 (M5-PA vs M5-PC vs M5-NoNeg)")
    print("=" * 60)
    df_ep2, df_st2 = run_exp2(config, exp1_episodes_df=df_ep1)

    # 生成所有图表
    print("\n" + "=" * 60)
    print("生成可视化图表")
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
