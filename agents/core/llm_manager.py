from typing import Optional

from ..llm.base import BaseLLM


class LLMManager:
    _instance = None
    _global_llm: Optional[BaseLLM] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def set_global_llm(cls, llm: BaseLLM):
        cls._global_llm = llm

    @classmethod
    def get_global_llm(cls) -> Optional[BaseLLM]:
        return cls._global_llm
