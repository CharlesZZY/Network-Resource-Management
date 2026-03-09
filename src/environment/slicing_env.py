import gymnasium as gym
import numpy as np
from gymnasium import spaces


# 3GPP CQI到频谱效率映射 (bits/s/Hz)
CQI_TO_SE = {
    1: 0.15, 2: 0.23, 3: 0.38, 4: 0.60, 5: 0.88,
    6: 1.18, 7: 1.48, 8: 1.91, 9: 2.41, 10: 2.73,
    11: 3.32, 12: 3.90, 13: 4.52, 14: 5.12, 15: 5.55,
}


class NetworkSlicingEnv(gym.Env):
    """5G网络切片资源管理仿真环境"""

    metadata = {"render_modes": []}

    TECH_MULTIPLIER = 20      # MIMO+编码增益
    EMBB_PACKET_WEIGHT = 10   # eMBB大数据包权重(用于时延计算)
    URLLC_PACKET_WEIGHT = 0.4 # URLLC小数据包权重
    MMTC_ACCESS_FACTOR = 1.0  # mMTC接入容量系数

    def __init__(self, config: dict, scenario: str = "steady"):
        super().__init__()
        self.cfg = config["environment"]
        self.sla_cfg = config["sla"]
        self.reward_cfg = config["reward"]
        self.scenario = scenario

        self.total_bw = self.cfg["total_bandwidth_mhz"]
        self.max_users = self.cfg["max_users"]
        self.num_slices = self.cfg["num_slices"]
        self.arrival_rates = np.array(self.cfg["arrival_rates"], dtype=np.float64)
        self.min_alloc = self.cfg["min_allocation"]
        self.burst_cycle = self.cfg.get("burst_cycle", 50)
        self.burst_multiplier = self.cfg.get("burst_multiplier", 2.0)

        # CQI AR(1) 均值回复模型参数
        self._cqi_means = np.array(self.cfg.get("cqi_means", [9, 10, 7]), dtype=np.float64)
        self._cqi_alpha = self.cfg.get("cqi_reversion_rate", 0.15)
        self._cqi_sigma = self.cfg.get("cqi_noise_std", 1.2)

        self.observation_space = spaces.Box(0.0, 1.0, shape=(15,), dtype=np.float64)
        self.action_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float64)

        self.current_step = 0
        self.rng = np.random.default_rng(42)
        self._sla_window = 10
        self._sla_history = [[] for _ in range(self.num_slices)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.current_step = 0

        self._users = np.array([
            self.rng.poisson(lam) for lam in self.arrival_rates
        ], dtype=np.float64)
        self._users = np.clip(self._users, 1, self.max_users)

        self._cqi = self._cqi_means + self.rng.normal(0, self._cqi_sigma, size=self.num_slices)
        self._cqi = np.clip(np.round(self._cqi), 1, 15).astype(np.float64)
        self._buffer = self.rng.uniform(0.1, 0.3, size=self.num_slices)
        self._allocation = np.array([0.5, 0.3, 0.2])
        self._sla_rates = np.ones(self.num_slices)
        self._sla_history = [[] for _ in range(self.num_slices)]

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = self._normalize_action(action)
        self.current_step += 1
        self._allocation = action

        self._update_users()
        self._update_cqi()
        self._update_buffers(action)

        sla_met, metrics = self._evaluate_sla(action)
        for i in range(self.num_slices):
            self._sla_history[i].append(float(sla_met[i]))
            if len(self._sla_history[i]) > self._sla_window:
                self._sla_history[i] = self._sla_history[i][-self._sla_window:]
            self._sla_rates[i] = np.mean(self._sla_history[i])

        reward = self._compute_reward(sla_met, action, metrics)
        terminated = self.current_step >= self.cfg.get("max_steps", 100)

        info = self._get_info()
        info.update(metrics)
        info["sla_met"] = sla_met.tolist()

        return self._get_obs(), reward, terminated, False, info

    def _normalize_action(self, action):
        action = np.clip(action, 0.0, 1.0)
        action = np.maximum(action, self.min_alloc)
        action = action / action.sum()
        return action

    def _update_users(self):
        rates = self.arrival_rates.copy()
        if self.scenario == "burst" and self.current_step >= self.burst_cycle:
            rates[0] *= self.burst_multiplier
        self._users = np.array([
            self.rng.poisson(lam) for lam in rates
        ], dtype=np.float64)
        self._users = np.clip(self._users, 1, self.max_users)

    def _update_cqi(self):
        # 离散 Ornstein-Uhlenbeck (AR(1)均值回复):
        # CQI(t+1) = CQI(t) + α*(μ - CQI(t)) + σ*ε
        drift = self._cqi_alpha * (self._cqi_means - self._cqi)
        noise = self.rng.normal(0, self._cqi_sigma, size=self.num_slices)
        self._cqi = np.clip(np.round(self._cqi + drift + noise), 1, 15)

    def _update_buffers(self, action):
        load_factor = self._users / (self.max_users * action + 1e-6)
        target_buffer = np.clip(load_factor / 5.0, 0.0, 1.0)
        self._buffer = 0.7 * self._buffer + 0.3 * target_buffer
        noise = self.rng.normal(0, 0.02, size=self.num_slices)
        self._buffer = np.clip(self._buffer + noise, 0.0, 1.0)

    def _evaluate_sla(self, action):
        sla_met = np.zeros(self.num_slices, dtype=bool)
        metrics = {}
        M = self.TECH_MULTIPLIER

        # --- eMBB: 平均用户吞吐量 >= 50 Mbps ---
        embb_bw = action[0] * self.total_bw
        embb_se = CQI_TO_SE[int(self._cqi[0])]
        embb_cap_per_user = (embb_bw * embb_se * M) / max(self._users[0], 1)
        sla_met[0] = embb_cap_per_user >= self.sla_cfg["embb_throughput_mbps"]
        metrics["embb_throughput"] = embb_cap_per_user

        # --- URLLC: 基于数据包大小差异的相对时延模型 ---
        # 时延 ∝ packet_weight / per_user_capacity
        embb_delay = self.EMBB_PACKET_WEIGHT / max(embb_cap_per_user, 0.01)

        urllc_bw = action[1] * self.total_bw
        urllc_se = CQI_TO_SE[int(self._cqi[1])]
        urllc_cap_per_user = (urllc_bw * urllc_se * M) / max(self._users[1], 1)
        urllc_avg_delay = self.URLLC_PACKET_WEIGHT / max(urllc_cap_per_user, 0.01)
        urllc_p99_delay = urllc_avg_delay * 2.3

        sla_met[1] = urllc_p99_delay <= self.sla_cfg["urllc_delay_ratio"] * embb_delay
        metrics["urllc_p99_delay"] = urllc_p99_delay
        metrics["embb_delay"] = embb_delay

        # --- mMTC: 接入成功率 >= 95% ---
        mmtc_bw = action[2] * self.total_bw
        mmtc_se = CQI_TO_SE[int(self._cqi[2])]
        mmtc_capacity = mmtc_bw * mmtc_se * self.MMTC_ACCESS_FACTOR
        access_rate = min(1.0, mmtc_capacity / max(self._users[2], 0.01))
        sla_met[2] = access_rate >= self.sla_cfg["mMTC_access_rate"]
        metrics["mmtc_access_rate"] = access_rate

        # 带宽利用率: 基于实际负载与总容量比
        total_demand = (
            self._users[0] * 20.0 + self._users[1] * 2.0 + self._users[2] * 0.5
        ) / M
        metrics["bandwidth_util"] = min(1.0, total_demand / self.total_bw)

        # 各切片吞吐量汇总
        urllc_throughput = urllc_cap_per_user
        mmtc_bw_per_user = (mmtc_bw * mmtc_se * M) / max(self._users[2], 1)
        metrics["total_throughput"] = (
            embb_cap_per_user * self._users[0]
            + urllc_throughput * self._users[1]
            + mmtc_bw_per_user * self._users[2]
        )

        return sla_met, metrics

    def _compute_reward(self, sla_met, action, metrics):
        s_avg = np.mean(self._sla_rates)
        util = metrics["bandwidth_util"]
        violation = 1.0 - np.mean(sla_met.astype(float))
        reward = (
            self.reward_cfg["w_sla"] * s_avg
            + self.reward_cfg["w_util"] * util
            - self.reward_cfg["w_violation"] * violation
        )
        return reward

    def _get_obs(self):
        obs = np.concatenate([
            self._users / self.max_users,
            self._cqi / 15.0,
            self._buffer,
            self._allocation,
            self._sla_rates,
        ])
        return np.clip(obs, 0.0, 1.0)

    def _get_info(self):
        return {
            "step": self.current_step,
            "users": self._users.copy(),
            "cqi": self._cqi.copy(),
            "buffer": self._buffer.copy(),
            "allocation": self._allocation.copy(),
            "sla_rates": self._sla_rates.copy(),
        }

    def get_state_description(self):
        """将当前状态转为结构化自然语言描述, 供LLM Agent使用"""
        names = self.cfg["slice_names"]
        lines = [
            f"当前决策周期: {self.current_step}",
            f"系统总带宽: {self.total_bw} MHz",
        ]
        for i, name in enumerate(names):
            lines.append(
                f"[{name}] 用户数={int(self._users[i])}, "
                f"CQI={int(self._cqi[i])}, "
                f"缓冲区占用={self._buffer[i]:.2f}, "
                f"当前带宽分配={self._allocation[i]:.2f} ({self._allocation[i]*self.total_bw:.1f}MHz), "
                f"SLA满足率={self._sla_rates[i]:.2f}"
            )
        return "\n".join(lines)
