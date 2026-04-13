import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.environment.slicing_env import NetworkSlicingEnv
from src.utils.metrics import aggregate_episode_metrics


PER_METHOD_BASE = "results/data/per_method"


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Per-method result IO helpers
# ---------------------------------------------------------------------------

def per_method_dir(method_name: str, base: str = PER_METHOD_BASE) -> Path:
    """Return (and create) the directory for a given method's results."""
    d = Path(base) / method_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_method_results(method_name: str, ep_metrics: list[dict],
                        step_records: list[dict],
                        base: str = PER_METHOD_BASE) -> None:
    """Write a method's episode + step CSVs under results/data/per_method/<method>/."""
    d = per_method_dir(method_name, base)
    pd.DataFrame(ep_metrics).to_csv(d / "episodes.csv", index=False)
    pd.DataFrame(step_records).to_csv(d / "steps.csv", index=False)


def load_method_results(method_name: str,
                        base: str = PER_METHOD_BASE
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read back per-method CSVs. Raises FileNotFoundError if either file is missing."""
    d = Path(base) / method_name
    ep_path = d / "episodes.csv"
    st_path = d / "steps.csv"
    if not ep_path.exists() or not st_path.exists():
        raise FileNotFoundError(
            f"Per-method data for {method_name} not found under {d} "
            f"(missing episodes.csv or steps.csv)"
        )
    return pd.read_csv(ep_path), pd.read_csv(st_path)


SUMMARY_METRICS = [
    "sla_embb", "sla_urllc", "sla_mmtc", "sla_avg",
    "throughput_mean", "bandwidth_util", "fairness",
    "decision_latency_mean", "total_tokens",
]


def build_summary_table(df_episodes: pd.DataFrame) -> pd.DataFrame:
    """Group by (method, scenario) and compute mean/std for the summary columns."""
    agg = df_episodes.groupby(["method", "scenario"]).agg(
        {m: ["mean", "std"] for m in SUMMARY_METRICS}
    ).round(4)
    agg.columns = [f"{col[0]}_{col[1]}" for col in agg.columns]
    return agg


def run_episode(env: NetworkSlicingEnv, policy, num_cycles: int, seed: int,
                progress_desc: str = ""):
    obs, info = env.reset(seed=seed)
    step_records = []

    pbar = tqdm(range(num_cycles), desc=progress_desc, leave=True,
                ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for step in pbar:
        action, latency, tokens = policy.decide(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)

        sla_met = info.get("sla_met", [False, False, False])
        alloc = info.get("allocation", np.array([0.5, 0.3, 0.2]))
        sla_r = info.get("sla_rates", np.zeros(3))
        if hasattr(sla_r, "tolist"):
            sla_r = sla_r.tolist()
        if hasattr(alloc, "tolist"):
            alloc = alloc.tolist()

        record = {
            "step": step,
            "sla_met": sla_met,
            "sla_embb_step": int(sla_met[0]),
            "sla_urllc_step": int(sla_met[1]),
            "sla_mmtc_step": int(sla_met[2]),
            "sla_avg_step": np.mean(sla_met),
            "total_throughput": info.get("total_throughput", 0),
            "bandwidth_util": info.get("bandwidth_util", 0),
            "decision_latency": latency,
            "token_usage": tokens,
            "reward": reward,
            "alloc_embb": alloc[0],
            "alloc_urllc": alloc[1],
            "alloc_mmtc": alloc[2],
            "sla_rate_embb": sla_r[0],
            "sla_rate_urllc": sla_r[1],
            "sla_rate_mmtc": sla_r[2],
        }
        step_records.append(record)

        if terminated or truncated:
            pbar.update(num_cycles - step - 1)
            break

    pbar.close()
    return step_records


def run_experiment(policy_factory, config: dict, scenario: str,
                   method_name: str, num_seeds: int = None, seeds: list = None):
    exp_cfg = config["experiment"]
    seeds = seeds or exp_cfg["seeds"]
    num_seeds = num_seeds or exp_cfg["num_seeds"]
    seeds = seeds[:num_seeds]
    num_cycles = exp_cfg["num_cycles"]

    all_episode_metrics = []
    all_step_records = []

    for i, seed in enumerate(seeds):
        env = NetworkSlicingEnv(config, scenario=scenario)
        env.cfg["max_steps"] = num_cycles
        policy = policy_factory()

        if hasattr(policy, "reset"):
            policy.reset()

        desc = f"  {method_name} | {scenario}/seed={seed} ({i+1}/{len(seeds)})"
        step_records = run_episode(env, policy, num_cycles, seed,
                                   progress_desc=desc)

        episode_metrics = aggregate_episode_metrics(step_records)
        episode_metrics["seed"] = seed
        episode_metrics["scenario"] = scenario
        episode_metrics["method"] = method_name
        episode_metrics["total_time"] = sum(r["decision_latency"] for r in step_records)

        all_episode_metrics.append(episode_metrics)

        for r in step_records:
            r["seed"] = seed
            r["scenario"] = scenario
            r["method"] = method_name
        all_step_records.extend(step_records)

    return all_episode_metrics, all_step_records
