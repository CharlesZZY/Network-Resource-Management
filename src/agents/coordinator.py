from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agents.llm_client import get_llm

COORDINATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是5G网络切片全局协调智能体,负责跨切片资源协调。\n\n"
     "职责:\n"
     "1. 收集各切片Agent的资源需求提案\n"
     "2. 评估全局资源约束(总带宽分配比例之和=1.0, 每切片最低0.05)\n"
     "3. 判断提案是否兼容(总需求是否超过可用资源)\n"
     "4. 如存在冲突,提出分配方案\n\n"
     "优先级: URLLC > eMBB > mMTC\n\n"
     "输出JSON格式(不要包含其他内容):\n"
     '{{"compatible": true/false, '
     '"allocation": {{"eMBB": 0.xx, "URLLC": 0.xx, "mMTC": 0.xx}}, '
     '"feedback": "分配理由或给各切片的反馈"}}'),
    ("human",
     "当前网络状态:\n{state_description}\n\n"
     "各切片提案:\n{proposals_text}\n\n"
     "总请求带宽比例: {total_requested:.2f} (可用: 1.00)\n\n"
     "请评估提案兼容性并给出分配方案。"),
])


class CoordinatorAgent:
    """全局协调智能体: PromptTemplate → LLM → JsonOutputParser"""

    def __init__(self):
        self.parser = JsonOutputParser()
        llm = get_llm()
        self.chain = COORDINATOR_PROMPT | llm | self.parser

    def evaluate_proposals(self, proposals: list[dict],
                           state_description: str) -> tuple[dict, int]:
        proposals_text = "\n".join([
            f"[{p['slice']}] 请求={p['requested']:.2f}, "
            f"最低={p['minimum']:.2f}, 优先级={p['priority']}, "
            f"理由={p.get('justification', 'N/A')}"
            for p in proposals
        ])
        total_requested = sum(p["requested"] for p in proposals)

        try:
            result = self.chain.invoke({
                "state_description": state_description,
                "proposals_text": proposals_text,
                "total_requested": total_requested,
            })
            evaluation = self._validate(result, proposals)
        except Exception:
            evaluation = self._fallback_allocation(proposals)

        tokens = len(state_description + proposals_text) // 2 + 200
        return evaluation, tokens

    def _validate(self, parsed: dict, proposals: list[dict]) -> dict:
        if "allocation" not in parsed:
            return self._fallback_allocation(proposals)
        alloc = parsed["allocation"]
        allocation = {
            "eMBB": float(alloc.get("eMBB", 0.5)),
            "URLLC": float(alloc.get("URLLC", 0.3)),
            "mMTC": float(alloc.get("mMTC", 0.2)),
        }
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v / total for k, v in allocation.items()}
        return {
            "compatible": parsed.get("compatible", False),
            "allocation": allocation,
            "feedback": parsed.get("feedback", ""),
        }

    def _fallback_allocation(self, proposals: list[dict]) -> dict:
        total = sum(p["requested"] for p in proposals) or 1.0
        allocation = {p["slice"]: max(p["requested"] / total, 0.05) for p in proposals}
        s = sum(allocation.values())
        allocation = {k: v / s for k, v in allocation.items()}
        return {
            "compatible": total <= 1.0,
            "allocation": allocation,
            "feedback": "回退: 按请求比例归一化",
        }
