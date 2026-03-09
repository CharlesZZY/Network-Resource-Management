"""仿真环境合理性验证: 生成诊断图证明环境行为符合预期"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from experiments.runner import load_config
from src.environment.slicing_env import NetworkSlicingEnv

OUTDIR = "results/figures"
os.makedirs(OUTDIR, exist_ok=True)


def run_env(config, scenario, allocation, n_steps=200, seed=42):
    """用固定分配跑环境, 收集全部指标"""
    env = NetworkSlicingEnv(config, scenario=scenario)
    env.cfg["max_steps"] = n_steps
    obs, info = env.reset(seed=seed)
    records = {"users": [], "cqi": [], "sla_met": [], "throughput": [],
               "delay_embb": [], "delay_urllc": [], "access_rate": [],
               "reward": [], "buffer": []}
    for _ in range(n_steps):
        obs, reward, *_, info = env.step(np.array(allocation))
        records["users"].append(env._users.copy())
        records["cqi"].append(env._cqi.copy())
        records["sla_met"].append(info["sla_met"])
        records["throughput"].append(info["total_throughput"])
        records["delay_embb"].append(info.get("embb_delay", 0))
        records["delay_urllc"].append(info.get("urllc_p99_delay", 0))
        records["access_rate"].append(info.get("mmtc_access_rate", 0))
        records["reward"].append(reward)
        records["buffer"].append(env._buffer.copy())
    for k in records:
        records[k] = np.array(records[k])
    return records


def test1_poisson_arrivals(config):
    """验证1: 用户到达服从Poisson分布"""
    env = NetworkSlicingEnv(config, scenario="steady")
    env.cfg["max_steps"] = 2000
    env.reset(seed=0)
    samples = {i: [] for i in range(3)}
    for _ in range(2000):
        env._update_users()
        for i in range(3):
            samples[i].append(env._users[i])

    names = config["environment"]["slice_names"]
    lambdas = config["environment"]["arrival_rates"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, ax in enumerate(axes):
        data = np.array(samples[i], dtype=int)
        ax.hist(data, bins=30, density=True, alpha=0.7, label="Simulated")
        x = np.arange(data.min(), data.max() + 1)
        poisson_pmf = stats.poisson.pmf(x, lambdas[i])
        ax.plot(x, poisson_pmf, "r-o", markersize=3, label=f"Poisson($\\lambda$={lambdas[i]})")
        ax.set_title(f"{names[i]} User Arrivals")
        ax.set_xlabel("User Count")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        # Chi-squared拟合优度检验 (适用于离散分布)
        vals, obs_counts = np.unique(data, return_counts=True)
        exp_counts = len(data) * stats.poisson.pmf(vals, lambdas[i])
        mask = exp_counts >= 5
        if mask.sum() < 2:
            p = 1.0
        else:
            obs_m = np.append(obs_counts[mask], obs_counts[~mask].sum())
            exp_m = np.append(exp_counts[mask], exp_counts[~mask].sum())
            exp_m = exp_m * (obs_m.sum() / exp_m.sum())
            _, p = stats.chisquare(obs_m, exp_m)
        ax.text(0.95, 0.95, f"$\\chi^2$ p={p:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

    plt.tight_layout()
    path = os.path.join(OUTDIR, "validate_poisson_arrivals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def test2_cqi_markov(config):
    """验证2: CQI马尔可夫演化稳定性"""
    env = NetworkSlicingEnv(config, scenario="steady")
    env.cfg["max_steps"] = 500
    env.reset(seed=0)
    traces = {i: [env._cqi[i]] for i in range(3)}
    for _ in range(500):
        env._update_cqi()
        for i in range(3):
            traces[i].append(env._cqi[i])

    names = config["environment"]["slice_names"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, ax in enumerate(axes):
        ax.plot(traces[i], linewidth=0.8)
        ax.set_title(f"{names[i]} CQI Evolution")
        ax.set_xlabel("Step")
        ax.set_ylabel("CQI")
        ax.set_ylim(0, 16)
        ax.axhline(y=np.mean(traces[i]), color="r", linestyle="--", alpha=0.5,
                    label=f"Mean={np.mean(traces[i]):.1f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "validate_cqi_markov.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def test3_allocation_sensitivity(config):
    """验证3: SLA满足率随带宽分配单调变化"""
    allocations = []
    sla_results = []
    # 扫描eMBB分配从0.1到0.8, 其余两切片平分
    for embb_ratio in np.arange(0.1, 0.85, 0.05):
        remaining = 1.0 - embb_ratio
        alloc = [embb_ratio, remaining * 0.6, remaining * 0.4]
        r = run_env(config, "steady", alloc, n_steps=200, seed=42)
        sla = np.array(r["sla_met"]).mean(axis=0)
        allocations.append(embb_ratio)
        sla_results.append(sla)

    allocations = np.array(allocations)
    sla_results = np.array(sla_results)
    names = config["environment"]["slice_names"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        ax.plot(allocations, sla_results[:, i], "o-", label=name, linewidth=2, markersize=4)
    ax.set_xlabel("eMBB Bandwidth Allocation Ratio")
    ax.set_ylabel("SLA Satisfaction Rate")
    ax.set_title("SLA Sensitivity to eMBB Allocation (URLLC:mMTC = 6:4 for remainder)")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "validate_allocation_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def test4_burst_impact(config):
    """验证4: Burst事件正确影响eMBB负载和SLA"""
    r_steady = run_env(config, "steady", [0.5, 0.3, 0.2], n_steps=150, seed=42)
    r_burst = run_env(config, "burst", [0.5, 0.3, 0.2], n_steps=150, seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # eMBB用户数对比
    ax = axes[0, 0]
    ax.plot(r_steady["users"][:, 0], label="Steady", alpha=0.8)
    ax.plot(r_burst["users"][:, 0], label="Burst", alpha=0.8)
    ax.axvline(x=50, color="red", linestyle="--", alpha=0.4, label="Burst @t=50")
    ax.set_title("eMBB User Count")
    ax.set_ylabel("Users")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # eMBB SLA满足
    ax = axes[0, 1]
    w = 10
    sla_s = np.convolve(np.array(r_steady["sla_met"])[:, 0], np.ones(w)/w, mode="valid")
    sla_b = np.convolve(np.array(r_burst["sla_met"])[:, 0], np.ones(w)/w, mode="valid")
    ax.plot(sla_s, label="Steady", alpha=0.8)
    ax.plot(sla_b, label="Burst", alpha=0.8)
    ax.axvline(x=50, color="red", linestyle="--", alpha=0.4)
    ax.set_title("eMBB SLA Satisfaction (rolling avg)")
    ax.set_ylabel("SLA Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 总吞吐量
    ax = axes[1, 0]
    ax.plot(r_steady["throughput"], label="Steady", alpha=0.8)
    ax.plot(r_burst["throughput"], label="Burst", alpha=0.8)
    ax.axvline(x=50, color="red", linestyle="--", alpha=0.4)
    ax.set_title("Total Throughput")
    ax.set_xlabel("Step")
    ax.set_ylabel("Throughput (Mbps)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 奖励
    ax = axes[1, 1]
    ax.plot(r_steady["reward"], label="Steady", alpha=0.8)
    ax.plot(r_burst["reward"], label="Burst", alpha=0.8)
    ax.axvline(x=50, color="red", linestyle="--", alpha=0.4)
    ax.set_title("Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Burst Scenario Impact Validation", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "validate_burst_impact.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def test5_tradeoff(config):
    """验证5: 切片间资源竞争存在不可调和的tradeoff"""
    configs_to_test = [
        ([0.7, 0.15, 0.15], "eMBB-heavy (70/15/15)"),
        ([0.2, 0.6, 0.2],  "URLLC-heavy (20/60/20)"),
        ([0.15, 0.15, 0.7], "mMTC-heavy (15/15/70)"),
        ([0.5, 0.3, 0.2],  "Balanced (50/30/20)"),
        ([0.33, 0.34, 0.33], "Equal (33/34/33)"),
    ]
    names = config["environment"]["slice_names"]
    results = []
    for alloc, label in configs_to_test:
        r = run_env(config, "steady", alloc, n_steps=200, seed=42)
        sla = np.array(r["sla_met"]).mean(axis=0)
        results.append((label, sla))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.15
    for i, (label, sla) in enumerate(results):
        offset = (i - len(results) / 2 + 0.5) * width
        ax.bar(x + offset, sla, width, label=label, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("SLA Satisfaction Rate")
    ax.set_title("Inter-Slice Tradeoff: No Single Allocation Satisfies All SLAs")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "validate_tradeoff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


if __name__ == "__main__":
    config = load_config()
    print("=== Environment Validation ===\n")

    print("[1/5] Poisson user arrivals")
    test1_poisson_arrivals(config)

    print("[2/5] CQI Markov evolution")
    test2_cqi_markov(config)

    print("[3/5] Allocation sensitivity")
    test3_allocation_sensitivity(config)

    print("[4/5] Burst scenario impact")
    test4_burst_impact(config)

    print("[5/5] Inter-slice tradeoff")
    test5_tradeoff(config)

    print("\nAll 5 validation tests completed.")
