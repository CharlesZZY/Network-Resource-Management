from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agents.llm_client import get_llm

SLICE_SYSTEM_PROMPTS = {
    "eMBB": (
        "你是eMBB切片管理智能体,负责高带宽移动宽带服务(视频流、大文件下载)。\n"
        "SLA要求: 平均用户吞吐量 >= 50 Mbps。"
    ),
    "URLLC": (
        "你是URLLC切片管理智能体,负责超可靠低时延通信(工业控制、远程手术)。\n"
        "SLA要求: 99th百分位排队时延 <= eMBB时延的10%。URLLC拥有最高优先级。"
    ),
    "mMTC": (
        "你是mMTC切片管理智能体,负责海量机器类通信(IoT传感器网络)。\n"
        "SLA要求: 接入成功率 >= 95%。"
    ),
}

PROPOSAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "{system_prompt}\n\n"
     "请用Chain-of-Thought推理分析当前切片状态,然后输出JSON格式的资源需求提案。\n"
     "输出JSON格式(不要包含其他内容):\n"
     '{{"requested": 0.xx, "minimum": 0.xx, "priority": "high/medium/low", '
     '"justification": "简要理由"}}'),
    ("human",
     "当前网络状态:\n{state_description}\n\n请为{slice_name}切片提出资源需求提案。"),
])

COUNTER_PROPOSAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "{system_prompt}\n\n"
     "请根据协调者反馈调整你的资源需求提案。输出JSON格式(不要包含其他内容):\n"
     '{{"requested": 0.xx, "minimum": 0.xx, "priority": "high/medium/low", '
     '"justification": "简要理由"}}'),
    ("human",
     "当前网络状态:\n{state_description}\n\n"
     "协调者反馈:\n{coordinator_feedback}\n\n"
     "请调整你的{slice_name}切片资源需求提案。"),
])

DEFAULTS = {
    "eMBB": {"requested": 0.50, "minimum": 0.30, "priority": "medium", "justification": "default"},
    "URLLC": {"requested": 0.35, "minimum": 0.25, "priority": "high", "justification": "default"},
    "mMTC": {"requested": 0.25, "minimum": 0.10, "priority": "low", "justification": "default"},
}


class SliceAgent:
    """切片管理智能体: PromptTemplate → LLM → JsonOutputParser"""

    def __init__(self, slice_name: str):
        self.slice_name = slice_name
        self.system_prompt = SLICE_SYSTEM_PROMPTS[slice_name]
        self.parser = JsonOutputParser()
        llm = get_llm()
        self.proposal_chain = PROPOSAL_PROMPT | llm | self.parser
        self.counter_chain = COUNTER_PROPOSAL_PROMPT | llm | self.parser

    def generate_proposal(self, state_description: str) -> tuple[dict, int]:
        try:
            result = self.proposal_chain.invoke({
                "system_prompt": self.system_prompt,
                "state_description": state_description,
                "slice_name": self.slice_name,
            })
            proposal = self._validate(result)
        except Exception:
            proposal = dict(DEFAULTS[self.slice_name])
        proposal["slice"] = self.slice_name
        tokens = self._estimate_tokens(state_description)
        return proposal, tokens

    def generate_counter_proposal(self, state_description: str,
                                  coordinator_feedback: str) -> tuple[dict, int]:
        try:
            result = self.counter_chain.invoke({
                "system_prompt": self.system_prompt,
                "state_description": state_description,
                "coordinator_feedback": coordinator_feedback,
                "slice_name": self.slice_name,
            })
            proposal = self._validate(result)
        except Exception:
            proposal = dict(DEFAULTS[self.slice_name])
        proposal["slice"] = self.slice_name
        tokens = self._estimate_tokens(state_description + coordinator_feedback)
        return proposal, tokens

    def _validate(self, parsed: dict) -> dict:
        return {
            "requested": float(parsed.get("requested", 0.3)),
            "minimum": max(float(parsed.get("minimum", 0.1)), 0.05),
            "priority": parsed.get("priority", "medium"),
            "justification": parsed.get("justification", ""),
        }

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 2 + 150
