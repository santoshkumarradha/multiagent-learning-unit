from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from pydantic import BaseModel


class BaseLLM(ABC):
    def __init__(
        self, temperature: float = 1.0, top_p: float = 1.0, max_tokens: int = 1500
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(
        self, system_prompt: str, context: str, task: str, schema: Type[BaseModel]
    ) -> str:
        pass
