from .config import GlobalConfig
from .core.agent import Agent
from .core.llm_manager import LLMManager
from .llm import BaseLLM, MockLLM, OpenAIChatGPT


def set_global_llm(llm: BaseLLM):
    LLMManager.set_global_llm(llm)


def set_live_verbosity(level: int):
    GlobalConfig.LIVE_VERBOSITY_LEVEL = level
