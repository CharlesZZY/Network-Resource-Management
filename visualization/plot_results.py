"""可视化模块: 生成实验1和实验2的所有对比图"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib
matplotlib.use("Agg")


FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

METHOD_COLORS = {
    "M1-Fixed": "#7f8c8d",
    "M2-Threshold": "#3498db",
    "M4-SingleLLM": "#e67e22",
    "M5-PA": "#e74c3c",
    "M5-PC": "#2ecc71",
    "M5-NoNeg": "#9b59b6",
}

SLICE_NAMES = ["eMBB", "URLLC", "mMTC"]


def _sorted_methods(methods, color_map=METHOD_COLORS):
    order = list(color_map.keys())
    return sorted(methods, key=lambda m: order.index(m) if m in order else 99)


def _grouped_bar(ax, df, metric_col, methods, scenarios, ylabel, title,
                 scale=1.0):
    """通用分组柱状图: 按scenario分组, 每组内各method一根柱"""
    x = np.arange(len(scenarios))
    n = len(methods)
    width = 0.8 / max(n, 1)
    for i, method in enumerate(methods):
        m_data = df[df["method"] == method]
        means = [m_data[m_data["scenario"] == s][metric_col].mean() * scale
                 for s in scenarios]
        stds = [m_data[m_data["scenario"] == s][metric_col].std() * scale
                for s in scenarios]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=method, color=METHOD_COLORS.get(method, "#95a5a6"),
               capsize=3, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(axis="y", alpha=0.3)


# ===================== 实验1 图表 =====================

def plot_exp1_sla_comparison(df):
    """Exp1: 分切片SLA满足率柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sla_cols = ["sla_embb", "sla_urllc", "sla_mmtc"]
    scenarios = ["steady", "burst"]

    for ax, scenario in zip(axes, scenarios):
        sub = df[df["scenario"] == scenario]
        methods = _sorted_methods(sub["method"].unique())
        x = np.arange(len(SLICE_NAMES))
        n = len(methods)
        width = 0.8 / max(n, 1)
        for i, method in enumerate(methods):
            m_data = sub[sub["method"] == method]
            means = [m_data[c].mean() for c in sla_cols]
            stds = [m_data[c].std() for c in sla_cols]
            offset = (i - n / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=method, color=METHOD_COLORS.get(method, "#95a5a6"),
                   capsize=3, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Slice Type")
        ax.set_ylabel("SLA Satisfaction Rate")
        ax.set_title(f"Per-Slice SLA Satisfaction ({scenario.capitalize()})")
        ax.set_xticks(x)
        ax.set_xticklabels(SLICE_NAMES)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp1_sla_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_exp1_burst_trend(df_steps):
    """Exp1: Burst场景下SLA满足率时间曲线"""
    burst = df_steps[df_steps["scenario"] == "burst"]
    if burst.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    methods = _sorted_methods(burst["method"].unique())
    for method in methods:
        grouped = burst[burst["method"] == method].groupby("step")["sla_avg_step"].mean()
        smoothed = grouped.rolling(5, min_periods=1).mean()
        ax.plot(smoothed.index, smoothed.values, label=method,
                color=METHOD_COLORS.get(method, "#95a5a6"), linewidth=2, alpha=0.85)

    ax.axvline(x=50, color="red", linestyle="--", alpha=0.5, label="Burst Event")
    ax.set_xlabel("Decision Cycle")
    ax.set_ylabel("Average SLA Satisfaction Rate")
    ax.set_title("SLA Satisfaction Rate Over Time (Burst Scenario)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp1_burst_trend.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_exp1_performance_overview(df):
    """Exp1: 综合性能对比 (throughput, utilization, fairness, latency, cost)"""
    all_methods = _sorted_methods(df["method"].unique())
    llm_methods = [m for m in all_methods if df[df["method"] == m]["total_tokens"].sum() > 0]
    scenarios = ["steady", "burst"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    _grouped_bar(axes[0], df, "throughput_mean", all_methods, scenarios,
                 "Weighted Throughput (Mbps)", "Weighted Total Throughput")
    _grouped_bar(axes[1], df, "bandwidth_util", all_methods, scenarios,
                 "Bandwidth Utilization", "Bandwidth Utilization")
    _grouped_bar(axes[2], df, "fairness", all_methods, scenarios,
                 "Jain's Fairness Index", "Inter-Slice Fairness")
    _grouped_bar(axes[3], df, "decision_latency_mean", all_methods, scenarios,
                 "Latency (ms)", "Per-Decision Latency", scale=1000.0)
    _grouped_bar(axes[4], df, "total_tokens", llm_methods, scenarios,
                 "Tokens", "API Call Cost (tokens/episode)")
    axes[5].axis("off")

    fig.suptitle("Experiment 1: Comprehensive Performance Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp1_performance_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ===================== 实验2 图表 =====================

def plot_exp2_fairness_comparison(df):
    """Exp2: Jain公平性指数对比"""
    exp2 = ["M5-PA", "M5-PC", "M5-NoNeg"]
    sub = df[df["method"].isin(exp2)]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    _grouped_bar(ax, sub, "fairness", exp2, ["steady", "burst"],
                 "Jain's Fairness Index", "Inter-Slice Fairness (Experiment 2)")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp2_fairness_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_exp2_sla_per_slice(df):
    """Exp2: 各切片SLA满足率对比"""
    exp2 = ["M5-PA", "M5-PC", "M5-NoNeg"]
    sub = df[df["method"].isin(exp2)]
    if sub.empty:
        return

    sla_cols = ["sla_embb", "sla_urllc", "sla_mmtc"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scenarios = ["steady", "burst"]

    for ax, scenario in zip(axes, scenarios):
        s_data = sub[sub["scenario"] == scenario]
        x = np.arange(len(SLICE_NAMES))
        n = len(exp2)
        width = 0.8 / max(n, 1)
        for i, method in enumerate(exp2):
            m_data = s_data[s_data["method"] == method]
            means = [m_data[c].mean() for c in sla_cols]
            stds = [m_data[c].std() for c in sla_cols]
            offset = (i - n / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=method, color=METHOD_COLORS.get(method, "#95a5a6"),
                   capsize=3, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Slice Type")
        ax.set_ylabel("SLA Satisfaction Rate")
        ax.set_title(f"Per-Slice SLA ({scenario.capitalize()}) - Exp2")
        ax.set_xticks(x)
        ax.set_xticklabels(SLICE_NAMES)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp2_sla_per_slice.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_exp2_performance_overview(df):
    """Exp2: 综合性能对比 (throughput, utilization, latency, cost)"""
    exp2 = ["M5-PA", "M5-PC", "M5-NoNeg"]
    sub = df[df["method"].isin(exp2)]
    if sub.empty:
        return

    scenarios = ["steady", "burst"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    _grouped_bar(axes[0], sub, "throughput_mean", exp2, scenarios,
                 "Weighted Throughput (Mbps)", "Weighted Total Throughput")
    _grouped_bar(axes[1], sub, "bandwidth_util", exp2, scenarios,
                 "Bandwidth Utilization", "Bandwidth Utilization")
    _grouped_bar(axes[2], sub, "decision_latency_mean", exp2, scenarios,
                 "Latency (ms)", "Per-Decision Latency", scale=1000.0)
    _grouped_bar(axes[3], sub, "total_tokens", exp2, scenarios,
                 "Tokens", "API Call Cost (tokens/episode)")

    fig.suptitle("Experiment 2: Negotiation Mechanism Performance", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp2_performance_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ===================== 入口 =====================

def generate_all_plots():
    print("\n=== Generating plots ===")

    if os.path.exists("results/data/exp1_episodes.csv"):
        df_ep1 = pd.read_csv("results/data/exp1_episodes.csv")
        plot_exp1_sla_comparison(df_ep1)
        plot_exp1_performance_overview(df_ep1)

    if os.path.exists("results/data/exp1_steps.csv"):
        df_st1 = pd.read_csv("results/data/exp1_steps.csv")
        plot_exp1_burst_trend(df_st1)

    if os.path.exists("results/data/exp2_episodes.csv"):
        df_ep2 = pd.read_csv("results/data/exp2_episodes.csv")
        plot_exp2_fairness_comparison(df_ep2)
        plot_exp2_sla_per_slice(df_ep2)
        plot_exp2_performance_overview(df_ep2)

    print("All plots generated.")


if __name__ == "__main__":
    generate_all_plots()
