"""实验1: 核心性能对比 (RQ1) — M1, M2, M4, M5-PA, M5-PC"""

import pandas as pd

from experiments.runner import load_config, run_experiment
from src.agents.multi_agent_graph import MultiAgentPolicy
from src.agents.single_agent import SingleAgentLLM
from src.baselines.fixed_ratio import FixedRatioPolicy
from src.baselines.threshold import ThresholdPolicy


def run_exp1(config=None):
    config = config or load_config()

    methods = {
        "M1-Fixed": lambda: FixedRatioPolicy(config["baseline"]["fixed_ratio"]),
        "M2-Threshold": lambda: ThresholdPolicy(config),
        "M4-SingleLLM": lambda: SingleAgentLLM(),
        "M5-PA": lambda: MultiAgentPolicy(strategy="priority", max_rounds=3),
        "M5-PC": lambda: MultiAgentPolicy(strategy="proportional", max_rounds=3),
    }

    all_episode_metrics = []
    all_step_records = []

    for scenario in config["experiment"]["scenarios"]:
        for name, factory in methods.items():
            print(f"\n=== 实验1: {name} / {scenario} ===")
            ep_metrics, step_records = run_experiment(
                factory, config, scenario, name
            )
            all_episode_metrics.extend(ep_metrics)
            all_step_records.extend(step_records)

    df_episodes = pd.DataFrame(all_episode_metrics)
    df_steps = pd.DataFrame(all_step_records)

    df_episodes.to_csv("results/data/exp1_episodes.csv", index=False)
    df_steps.to_csv("results/data/exp1_steps.csv", index=False)

    # 生成主表: 按method和scenario分组求均值和标准差
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
    summary.to_csv("results/data/exp1_main_table.csv")

    print("\n实验1完成, 数据已保存到 results/data/")
    return df_episodes, df_steps


if __name__ == "__main__":
    run_exp1()
