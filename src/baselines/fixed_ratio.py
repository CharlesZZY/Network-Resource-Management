import numpy as np


class FixedRatioPolicy:
    """M1: 固定比例分配 (eMBB:URLLC:mMTC = 5:3:2)"""

    def __init__(self, ratio=None):
        self.ratio = np.array(ratio or [0.5, 0.3, 0.2])

    def decide(self, obs, env=None):
        return self.ratio.copy(), 0, 0
