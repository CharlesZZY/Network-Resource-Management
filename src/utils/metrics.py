import numpy as np


def jain_fairness_index(values):
    """Jain公平性指数: J = (sum(x))^2 / (n * sum(x^2))"""
    values = np.array(values, dtype=np.float64)
    n = len(values)
    if n == 0 or np.sum(values ** 2) == 0:
        return 0.0
    return (np.sum(values) ** 2) / (n * np.sum(values ** 2))


def aggregate_episode_metrics(step_records: list[dict]) -> dict:
    """汇总单个episode所有step的指标"""
    n = len(step_records)
    if n == 0:
        return {}

    sla_met_arr = np.array([r["sla_met"] for r in step_records])
    per_slice_sla = sla_met_arr.mean(axis=0)

    throughputs = [r.get("total_throughput", 0) for r in step_records]
    utils = [r.get("bandwidth_util", 0) for r in step_records]
    latencies = [r.get("decision_latency", 0) for r in step_records]
    tokens = [r.get("token_usage", 0) for r in step_records]

    return {
        "sla_embb": per_slice_sla[0],
        "sla_urllc": per_slice_sla[1],
        "sla_mmtc": per_slice_sla[2],
        "sla_avg": per_slice_sla.mean(),
        "throughput_mean": np.mean(throughputs),
        "bandwidth_util": np.mean(utils),
        "fairness": jain_fairness_index(per_slice_sla),
        "decision_latency_mean": np.mean(latencies),
        "total_tokens": sum(tokens),
    }
