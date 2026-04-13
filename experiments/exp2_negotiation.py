"""实验2: 协商机制验证 (RQ2) — 单 method CLI 入口.

实验2涉及 M5-PA / M5-PC / M5-NoNeg 三种 method.
- M5-PA 和 M5-PC 与实验1共用 per-method 目录 (配置完全一致),
  通常在运行 exp1 时已写入; 此处仅需单独跑 M5-NoNeg.
- 也可以通过本脚本直接单独运行 M5-PA / M5-PC, 便于灵活并行.

每次调用只运行一种 method, 将结果写入
    results/data/per_method/<method>/{episodes.csv, steps.csv}

并行运行示例:
    python -m experiments.exp2_negotiation M5-NoNeg
"""

import argparse

from experiments.runner import (
    load_config,
    run_experiment,
    save_method_results,
)
from src.agents.multi_agent_graph import MultiAgentNoNegPolicy, MultiAgentPolicy


EXP2_METHODS = {
    "M5-PA":    lambda cfg: MultiAgentPolicy(strategy="priority",     max_rounds=3),
    "M5-PC":    lambda cfg: MultiAgentPolicy(strategy="proportional", max_rounds=3),
    "M5-NoNeg": lambda cfg: MultiAgentNoNegPolicy(),
}


def run_exp2_method(method_name: str, config: dict | None = None):
    """Run a single exp2 method across all scenarios, write per-method CSVs."""
    if method_name not in EXP2_METHODS:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Valid: {list(EXP2_METHODS)}"
        )

    config = config or load_config()
    factory_fn = EXP2_METHODS[method_name]

    all_episode_metrics = []
    all_step_records = []

    for scenario in config["experiment"]["scenarios"]:
        print(f"\n=== 实验2: {method_name} / {scenario} ===")
        ep_metrics, step_records = run_experiment(
            lambda: factory_fn(config), config, scenario, method_name
        )
        all_episode_metrics.extend(ep_metrics)
        all_step_records.extend(step_records)

    save_method_results(method_name, all_episode_metrics, all_step_records)
    print(f"\n实验2 ({method_name}) 完成, 数据已保存到 "
          f"results/data/per_method/{method_name}/")
    return all_episode_metrics, all_step_records


def main():
    parser = argparse.ArgumentParser(
        description="Run a single exp2 method and write per-method results."
    )
    parser.add_argument(
        "method",
        choices=list(EXP2_METHODS),
        help="Method id to run (M5-PA, M5-PC, M5-NoNeg).",
    )
    args = parser.parse_args()
    run_exp2_method(args.method)


if __name__ == "__main__":
    main()
