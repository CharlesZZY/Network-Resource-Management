import numpy as np


class ThresholdPolicy:
    """M2: 基于SLA违约阈值的动态调整策略"""

    def __init__(self, config: dict):
        self.threshold = config["baseline"]["threshold_sla"]
        self.transfer = config["baseline"]["threshold_transfer"]
        self.min_alloc = config["environment"]["min_allocation"]
        self.allocation = np.array(config["baseline"]["fixed_ratio"], dtype=np.float64)

    def decide(self, obs, env=None):
        sla_rates = obs[12:15]

        violated = np.where(sla_rates < self.threshold)[0]
        if len(violated) > 0:
            best = np.argmax(sla_rates)
            for v in violated:
                if v == best:
                    continue
                transfer = min(self.transfer, self.allocation[best] - self.min_alloc)
                if transfer > 0:
                    self.allocation[best] -= transfer
                    self.allocation[v] += transfer

        self.allocation = np.maximum(self.allocation, self.min_alloc)
        self.allocation /= self.allocation.sum()
        return self.allocation.copy(), 0, 0

    def reset(self, ratio=None):
        self.allocation = np.array(ratio or [0.5, 0.3, 0.2], dtype=np.float64)
