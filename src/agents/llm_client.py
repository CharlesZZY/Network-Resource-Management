import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

_llm_instance = None


def get_llm() -> ChatOpenAI:
    """获取全局共享的ChatOpenAI实例 (懒加载)"""
    global _llm_instance
    if _llm_instance is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("请在.env文件中设置OPENAI_API_KEY")
        _llm_instance = ChatOpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://aihubmix.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    return _llm_instance
