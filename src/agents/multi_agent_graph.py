import time
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from src.agents.coordinator import CoordinatorAgent
from src.agents.slice_agent import SliceAgent
from src.negotiation.protocol import (
    resolve_by_priority_arbitration,
    resolve_by_proportional_compromise,
)

SLICE_NAMES = ["eMBB", "URLLC", "mMTC"]


class MASState(TypedDict):
    state_description: str
    proposals: list[dict]
    coordinator_result: dict
    negotiation_round: int
    max_rounds: int
    strategy: str
    final_allocation: dict
    total_tokens: int
    is_resolved: bool


# ---- 节点构建函数 ----

def _build_slice_agents_node(agents: dict[str, SliceAgent]):
    def node(state: MASState) -> dict:
        proposals = []
        total_tokens = state.get("total_tokens", 0)
        desc = state["state_description"]
        round_num = state.get("negotiation_round", 0)
        feedback = state.get("coordinator_result", {}).get("feedback", "")

        for name in SLICE_NAMES:
            if round_num > 0 and feedback:
                proposal, tokens = agents[name].generate_counter_proposal(desc, feedback)
            else:
                proposal, tokens = agents[name].generate_proposal(desc)
            proposals.append(proposal)
            total_tokens += tokens

        return {"proposals": proposals, "total_tokens": total_tokens}
    return node


def _build_coordinator_node(coordinator: CoordinatorAgent):
    def node(state: MASState) -> dict:
        result, tokens = coordinator.evaluate_proposals(
            state["proposals"], state["state_description"]
        )
        return {
            "coordinator_result": result,
            "total_tokens": state.get("total_tokens", 0) + tokens,
        }
    return node


def _build_negotiation_node():
    def node(state: MASState) -> dict:
        proposals = state["proposals"]
        strategy = state.get("strategy", "priority")
        if strategy == "priority":
            allocation = resolve_by_priority_arbitration(proposals)
        else:
            allocation = resolve_by_proportional_compromise(proposals)
        return {
            "final_allocation": allocation,
            "negotiation_round": state.get("negotiation_round", 0) + 1,
            "is_resolved": True,
        }
    return node


def _build_finalize_node():
    def node(state: MASState) -> dict:
        if state.get("final_allocation"):
            return {"final_allocation": state["final_allocation"]}
        coord = state.get("coordinator_result", {})
        return {
            "final_allocation": coord.get(
                "allocation", {"eMBB": 0.5, "URLLC": 0.3, "mMTC": 0.2}
            )
        }
    return node


# ---- 条件路由函数 ----

def _should_negotiate(state: MASState) -> str:
    coord = state.get("coordinator_result", {})
    return "finalize" if coord.get("compatible", False) else "negotiation"


def _after_negotiation(state: MASState) -> str:
    if state.get("is_resolved") or state.get("negotiation_round", 0) >= state.get("max_rounds", 3):
        return "finalize"
    return "slice_agents"


# ---- 图构建 ----

def build_multi_agent_graph(strategy: str = "priority", max_rounds: int = 3):
    """构建LangGraph多智能体协作图 (含协商循环)"""
    slice_agents = {name: SliceAgent(name) for name in SLICE_NAMES}
    coordinator = CoordinatorAgent()

    graph = StateGraph(MASState)
    graph.add_node("slice_agents", _build_slice_agents_node(slice_agents))
    graph.add_node("coordinator", _build_coordinator_node(coordinator))
    graph.add_node("negotiation", _build_negotiation_node())
    graph.add_node("finalize", _build_finalize_node())

    graph.add_edge(START, "slice_agents")
    graph.add_edge("slice_agents", "coordinator")
    graph.add_conditional_edges("coordinator", _should_negotiate, {
        "finalize": "finalize",
        "negotiation": "negotiation",
    })
    graph.add_conditional_edges("negotiation", _after_negotiation, {
        "finalize": "finalize",
        "slice_agents": "slice_agents",
    })
    graph.add_edge("finalize", END)

    return graph.compile(), strategy, max_rounds


def build_no_negotiation_graph():
    """构建无协商变体图 (M5-no-neg)"""
    slice_agents = {name: SliceAgent(name) for name in SLICE_NAMES}
    coordinator = CoordinatorAgent()

    graph = StateGraph(MASState)
    graph.add_node("slice_agents", _build_slice_agents_node(slice_agents))
    graph.add_node("coordinator", _build_coordinator_node(coordinator))
    graph.add_node("finalize", _build_finalize_node())

    graph.add_edge(START, "slice_agents")
    graph.add_edge("slice_agents", "coordinator")
    graph.add_edge("coordinator", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# ---- 策略封装 ----

class MultiAgentPolicy:
    """M5 多智能体决策 (LangGraph编排, 含协商)"""

    def __init__(self, strategy: str = "priority", max_rounds: int = 3):
        self.graph, self.strategy, self.max_rounds = build_multi_agent_graph(
            strategy, max_rounds
        )
        self.fallback = np.array([0.5, 0.3, 0.2])

    def decide(self, obs, env=None):
        if env is None:
            return self.fallback.copy(), 0, 0

        initial_state: MASState = {
            "state_description": env.get_state_description(),
            "proposals": [],
            "coordinator_result": {},
            "negotiation_round": 0,
            "max_rounds": self.max_rounds,
            "strategy": self.strategy,
            "final_allocation": {},
            "total_tokens": 0,
            "is_resolved": False,
        }

        t0 = time.time()
        try:
            result = self.graph.invoke(initial_state)
            latency = time.time() - t0
            allocation = result.get("final_allocation", {})
            action = np.array([
                allocation.get("eMBB", 0.5),
                allocation.get("URLLC", 0.3),
                allocation.get("mMTC", 0.2),
            ])
            action = np.maximum(action, 0.05)
            action /= action.sum()
            return action, latency, result.get("total_tokens", 0)
        except Exception as e:
            print(f"[MultiAgent] 图执行失败: {e}")
            return self.fallback.copy(), time.time() - t0, 0


class MultiAgentNoNegPolicy:
    """M5-no-neg: 无协商多智能体 (Coordinator仍用LLM做一次性分配)"""

    def __init__(self):
        self.graph = build_no_negotiation_graph()
        self.fallback = np.array([0.5, 0.3, 0.2])

    def decide(self, obs, env=None):
        if env is None:
            return self.fallback.copy(), 0, 0

        initial_state: MASState = {
            "state_description": env.get_state_description(),
            "proposals": [],
            "coordinator_result": {},
            "negotiation_round": 0,
            "max_rounds": 0,
            "strategy": "none",
            "final_allocation": {},
            "total_tokens": 0,
            "is_resolved": False,
        }

        t0 = time.time()
        try:
            result = self.graph.invoke(initial_state)
            latency = time.time() - t0
            coord = result.get("coordinator_result", {})
            allocation = coord.get("allocation", {})
            action = np.array([
                allocation.get("eMBB", 0.5),
                allocation.get("URLLC", 0.3),
                allocation.get("mMTC", 0.2),
            ])
            action = np.maximum(action, 0.05)
            action /= action.sum()
            return action, latency, result.get("total_tokens", 0)
        except Exception as e:
            print(f"[MultiAgentNoNeg] 图执行失败: {e}")
            return self.fallback.copy(), time.time() - t0, 0
