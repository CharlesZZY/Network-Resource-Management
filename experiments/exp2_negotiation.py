"""实验2: 协商机制验证 (RQ2) — M5-PA vs M5-PC vs M5-no-neg"""

import os

import pandas as pd

from experiments.runner import load_config, run_experiment
from src.agents.multi_agent_graph import MultiAgentNoNegPolicy, MultiAgentPolicy


def run_exp2(config=None, exp1_episodes_df=None):
    config = config or load_config()

    # 尝试复用实验1中M5-PA和M5-PC的数据
    reused = []
    if exp1_episodes_df is not None:
        for m in ["M5-PA", "M5-PC"]:
            subset = exp1_episodes_df[exp1_episodes_df["method"] == m]
            if len(subset) > 0:
                reused.append(subset)
                print(f"复用实验1中 {m} 的数据 ({len(subset)} episodes)")

    # 如果没有可复用的, 重新跑M5-PA和M5-PC
    need_run = {}
    reused_methods = set()
    if reused:
        for df in reused:
            reused_methods.update(df["method"].unique())

    if "M5-PA" not in reused_methods:
        need_run["M5-PA"] = lambda: MultiAgentPolicy(strategy="priority", max_rounds=3)
    if "M5-PC" not in reused_methods:
        need_run["M5-PC"] = lambda: MultiAgentPolicy(strategy="proportional", max_rounds=3)
    need_run["M5-NoNeg"] = lambda: MultiAgentNoNegPolicy()

    all_episode_metrics = []
    all_step_records = []

    for scenario in config["experiment"]["scenarios"]:
        for name, factory in need_run.items():
            print(f"\n=== 实验2: {name} / {scenario} ===")
            ep_metrics, step_records = run_experiment(
                factory, config, scenario, name
            )
            all_episode_metrics.extend(ep_metrics)
            all_step_records.extend(step_records)

    df_new_episodes = pd.DataFrame(all_episode_metrics)
    df_new_steps = pd.DataFrame(all_step_records)

    # 合并复用数据
    if reused:
        df_episodes = pd.concat(reused + [df_new_episodes], ignore_index=True)
    else:
        df_episodes = df_new_episodes
    df_steps = df_new_steps  # step级数据只保留新运行的

    # 只保留实验2涉及的方法
    exp2_methods = ["M5-PA", "M5-PC", "M5-NoNeg"]
    df_episodes = df_episodes[df_episodes["method"].isin(exp2_methods)]

    df_episodes.to_csv("results/data/exp2_episodes.csv", index=False)
    df_steps.to_csv("results/data/exp2_steps.csv", index=False)

    summary = df_episodes.groupby(["method", "scenario"]).agg({
        "sla_embb": ["mean", "std"],
        "sla_urllc": ["mean", "std"],
        "sla_mmtc": ["mean", "std"],
        "sla_avg": ["mean", "std"],
        "throughput_mean": ["mean", "std"],
        "bandwidth_util": ["mean", "std"],
        "fairness": ["mean", "std"],
        "decision_latency_mean": ["mean", "std"],
        "total_tokens": ["mean", "std"],
    }).round(4)
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    summary.to_csv("results/data/exp2_comparison_table.csv")

    print("\n实验2完成, 数据已保存到 results/data/")
    return df_episodes, df_steps


if __name__ == "__main__":
    # 尝试加载实验1数据
    exp1_df = None
    if os.path.exists("results/data/exp1_episodes.csv"):
        exp1_df = pd.read_csv("results/data/exp1_episodes.csv")
    run_exp2(exp1_episodes_df=exp1_df)
