"""实验1: 核心性能对比 (RQ1) — 单 method CLI 入口.

每次调用只运行一种 method, 将结果写入
    results/data/per_method/<method>/{episodes.csv, steps.csv}

可在多个终端并行运行不同 method:
    python -m experiments.exp1_performance M1-Fixed
    python -m experiments.exp1_performance M3-PPO
    python -m experiments.exp1_performance M5-PA
"""

import argparse

from experiments.runner import (
    load_config,
    run_experiment,
    save_method_results,
)
from src.agents.multi_agent_graph import MultiAgentPolicy
from src.agents.single_agent import SingleAgentLLM
from src.baselines.fixed_ratio import FixedRatioPolicy
from src.baselines.ppo_policy import PPOPolicy
from src.baselines.threshold import ThresholdPolicy


EXP1_METHODS = {
    "M1-Fixed":     lambda cfg: FixedRatioPolicy(cfg["baseline"]["fixed_ratio"]),
    "M2-Threshold": lambda cfg: ThresholdPolicy(cfg),
    "M3-PPO":       lambda cfg: PPOPolicy(),
    "M4-SingleLLM": lambda cfg: SingleAgentLLM(),
    "M5-PA":        lambda cfg: MultiAgentPolicy(strategy="priority",     max_rounds=3),
    "M5-PC":        lambda cfg: MultiAgentPolicy(strategy="proportional", max_rounds=3),
}


def run_exp1_method(method_name: str, config: dict | None = None):
    """Run a single exp1 method across all scenarios, write per-method CSVs."""
    if method_name not in EXP1_METHODS:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Valid: {list(EXP1_METHODS)}"
        )

    config = config or load_config()
    factory_fn = EXP1_METHODS[method_name]

    all_episode_metrics = []
    all_step_records = []

    for scenario in config["experiment"]["scenarios"]:
        print(f"\n=== 实验1: {method_name} / {scenario} ===")
        ep_metrics, step_records = run_experiment(
            lambda: factory_fn(config), config, scenario, method_name
        )
        all_episode_metrics.extend(ep_metrics)
        all_step_records.extend(step_records)

    save_method_results(method_name, all_episode_metrics, all_step_records)
    print(f"\n实验1 ({method_name}) 完成, 数据已保存到 "
          f"results/data/per_method/{method_name}/")
    return all_episode_metrics, all_step_records


def main():
    parser = argparse.ArgumentParser(
        description="Run a single exp1 method and write per-method results."
    )
    parser.add_argument(
        "method",
        choices=list(EXP1_METHODS),
        help="Method id to run (e.g. M1-Fixed, M3-PPO, M5-PA).",
    )
    args = parser.parse_args()
    run_exp1_method(args.method)


if __name__ == "__main__":
    main()
