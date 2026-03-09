import time

import numpy as np
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agents.llm_client import get_llm

SINGLE_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个5G网络切片资源管理专家。根据当前网络状态,为三种切片分配带宽比例。\n\n"
     "SLA要求:\n"
     "- eMBB: 平均用户吞吐量 >= 50 Mbps\n"
     "- URLLC: 99th百分位排队时延 <= eMBB时延的10%\n"
     "- mMTC: 接入成功率 >= 95%\n\n"
     "约束: 三切片带宽比例之和=1.0, 每切片最低0.05, 优先级URLLC>eMBB>mMTC\n\n"
     "请用Chain-of-Thought推理分析后,输出JSON格式(不要包含其他内容):\n"
     '{{"eMBB": 0.xx, "URLLC": 0.xx, "mMTC": 0.xx}}'),
    ("human", "当前网络状态:\n{state_description}\n\n请给出带宽分配方案。"),
])


class SingleAgentLLM:
    """M4: 单智能体LLM — PromptTemplate → LLM → JsonOutputParser"""

    def __init__(self):
        self.parser = JsonOutputParser()
        llm = get_llm()
        self.chain = SINGLE_AGENT_PROMPT | llm | self.parser
        self.fallback = np.array([0.5, 0.3, 0.2])

    def decide(self, obs, env=None):
        if env is None:
            return self.fallback.copy(), 0, 0

        state_desc = env.get_state_description()
        t0 = time.time()
        try:
            result = self.chain.invoke({"state_description": state_desc})
            latency = time.time() - t0
            action = np.array([
                float(result.get("eMBB", 0.5)),
                float(result.get("URLLC", 0.3)),
                float(result.get("mMTC", 0.2)),
            ])
            action = np.maximum(action, 0.05)
            action /= action.sum()
            tokens = len(state_desc) // 2 + 150
            return action, latency, tokens
        except Exception:
            latency = time.time() - t0
            return self.fallback.copy(), latency, 0
