import time

import numpy as np
import yaml
from tqdm import tqdm

from src.environment.slicing_env import NetworkSlicingEnv
from src.utils.metrics import aggregate_episode_metrics


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


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
