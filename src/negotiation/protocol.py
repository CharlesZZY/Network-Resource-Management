import numpy as np

SLICE_NAMES = ["eMBB", "URLLC", "mMTC"]
# 优先级排序: URLLC(0) > eMBB(1) > mMTC(2)
PRIORITY_ORDER = [1, 0, 2]  # 索引对应 SLICE_NAMES


def resolve_by_priority_arbitration(proposals: list[dict]) -> dict:
    """M5-PA: 优先级仲裁 — URLLC优先满足,剩余按优先级分配"""
    allocation = {s: 0.05 for s in SLICE_NAMES}
    remaining = 1.0 - 0.05 * len(SLICE_NAMES)

    # 按优先级排序: URLLC > eMBB > mMTC
    sorted_proposals = sorted(
        proposals,
        key=lambda p: PRIORITY_ORDER.index(SLICE_NAMES.index(p["slice"]))
    )

    for p in sorted_proposals:
        name = p["slice"]
        requested = p["requested"] - 0.05  # 已预分配0.05
        give = min(max(requested, 0), remaining)
        allocation[name] += give
        remaining -= give

    # 剩余资源按比例分给未满足的切片
    if remaining > 0.001:
        for p in sorted_proposals:
            allocation[p["slice"]] += remaining / len(SLICE_NAMES)
        total = sum(allocation.values())
        allocation = {k: v / total for k, v in allocation.items()}

    return allocation


def resolve_by_proportional_compromise(proposals: list[dict]) -> dict:
    """M5-PC: 比例妥协 — 各切片按比例缩减,但不低于minimum"""
    total_requested = sum(p["requested"] for p in proposals)

    if total_requested <= 1.0:
        allocation = {p["slice"]: p["requested"] for p in proposals}
    else:
        # 先保障minimum, 剩余按比例分
        allocation = {p["slice"]: p["minimum"] for p in proposals}
        total_min = sum(allocation.values())
        remaining = max(1.0 - total_min, 0)
        excess_requests = {
            p["slice"]: max(p["requested"] - p["minimum"], 0) for p in proposals
        }
        total_excess = sum(excess_requests.values())
        if total_excess > 0:
            for p in proposals:
                share = excess_requests[p["slice"]] / total_excess * remaining
                allocation[p["slice"]] += share

    # 归一化确保sum=1
    total = sum(allocation.values())
    if total > 0:
        allocation = {k: max(v / total, 0.05) for k, v in allocation.items()}
        total = sum(allocation.values())
        allocation = {k: v / total for k, v in allocation.items()}

    return allocation
