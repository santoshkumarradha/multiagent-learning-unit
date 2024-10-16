import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

from .kmu import KMU

custom_theme = Theme(
    {
        "agent_name": "bold cyan",
        "task": "green",
        "response": "yellow",
    }
)


def format_agent_output(name: str, task: str, response: str):
    agent_info = f"[agent_name]{name}[/agent_name]\n"
    task_section = "[task]Task:[/task]\n" + task + "\n"
    response_section = "[response]Response:[/response]\n" + response

    full_content = agent_info + task_section + response_section
    return full_content


console = Console(theme=custom_theme, width=80)  #


# Base OperationalAgent class ---
class OperationalAgent:
    def generate_response(
        self, prompt: str, task: str, response_schema: Type[BaseModel]
    ) -> BaseModel:
        """
        Generate a response given a prompt and a response schema.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def validate_response(self, response: BaseModel, response_schema: Type[BaseModel]):
        """
        Validate the response to ensure it matches the expected response schema.
        """
        if not isinstance(response, response_schema):
            raise TypeError(f"Response must be of type {response_schema}")


# Default OperationalAgent using LLM ---
class DefaultOperationalAgent(OperationalAgent):
    def __init__(self, llm, verbose=False):
        self.llm = llm
        self.verbose = verbose

    def generate_response(
        self, prompt: str, task: str, response_schema: Type[BaseModel]
    ) -> BaseModel:
        # Generate response using LLM
        system_prompt = f"""You are OperationalAgent\nYour Role:Execute tasks based on the provided prompt\nYour Function:{prompt}"""
        result = self.llm.generate(
            self.llm.format_prompt(system_prompt=system_prompt, user_prompt=task),
            schema=response_schema,
        )

        if self.verbose:
            formatted_output = format_agent_output(
                "OperationalAgent", task, str(result)
            )
            console.print(
                Panel(
                    formatted_output,
                    title="OperationalAgent",
                    border_style="cyan",
                    expand=False,
                )
            )

        self.validate_response(result, response_schema)
        return result
